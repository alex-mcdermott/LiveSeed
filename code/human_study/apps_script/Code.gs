/**
 * LiveSeed human study: Google Apps Script web app bound to the study's Google Sheet.
 *
 * Endpoints (deploy: Deploy > New deployment > Web app; Execute as: Me; Who has access: Anyone):
 *   GET  ?action=assign&task=pairwise|calibration&pid=..&study=..&session=..[&pilot=1]
 *        -> {ok, batch_id} : the fullest session not yet full, so sessions fill in order and agreement can be read early (pilots: the
 *           least-filled session); every session needs RATERS clean completions. A slot counts as filled
 *           while its holder is active (assigned less than TIMEOUT_MIN ago) or completed and not excluded; excluded or timed-out
 *           slots are handed out again. Pilot participants get a session but never fill a slot.
 *   GET  ?action=export&key=EXPORT_KEY  -> every tab as JSON (for backups and for exporting the tabs to data/results/)
 *   GET  ?action=ping
 *   POST (Content-Type text/plain, JSON body; no CORS preflight)
 *        {action:"save",   task, pid, study, session, batch_id, rows:[...]}     answers as they are given
 *        {action:"submit", task, pid, study, session, batch_id, n_items, duration_sec, screen_w, screen_h, user_agent, pilot, rows:[...]}
 *            -> stores any rows not yet stored, checks that n_items distinct answers exist for (pid, batch), marks the assignment
 *               complete and returns {ok:true}; the page redirects to Prolific only after this.
 *        {action:"exclude", key, pids:[{pid, task}]}  marks assignments excluded so their slots are re-issued (scripts/03 --push)
 * Script Properties (Project Settings > Script properties): EXPORT_KEY (from secrets.yaml), S3_BASE
 *   (https://<bucket>.s3.<region>.amazonaws.com/), optional TIMEOUT_MIN_PAIRWISE (45) and TIMEOUT_MIN_CALIBRATION (60).
 * Run setup() once from the editor: it creates the tabs and loads the session lists from S3.
 */

// The tabs of the Sheet and their columns; scripts/common.py lists the same columns for the exported CSV files.
// "calibration" is the rating task of the study (paper App. E); "pairwise" is a second task that was not run.
var TABS = {
  batches: ["task", "batch_id", "raters"],
  assignments: ["timestamp", "pid", "study_id", "session_id", "task", "batch_id", "status", "pilot"],
  responses_pairwise: ["timestamp", "pid", "study_id", "session_id", "batch_id", "item_idx", "item_id", "left_img", "right_img",
                       "q_overall", "q_quality", "q_align", "rt_ms", "flagged"],
  responses_calibration: ["timestamp", "pid", "study_id", "session_id", "batch_id", "grid_idx", "grid_id", "img", "display_pos",
                          "pa", "iq", "rt_ms", "flagged"],
  responses_selection: ["timestamp", "pid", "study_id", "session_id", "batch_id", "grid_idx", "grid_id", "pick1", "pick2", "pick3", "shown", "rt_ms", "flagged"],
  submissions: ["timestamp", "pid", "study_id", "session_id", "task", "batch_id", "n_items", "duration_sec", "screen_w", "screen_h",
                "user_agent", "pilot"]
};

// ---------------- helpers
function props_() { return PropertiesService.getScriptProperties(); }
function sheet_(name) { return SpreadsheetApp.getActiveSpreadsheet().getSheetByName(name); }
function json_(obj) { return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON); }
// Reads a whole tab as a list of objects keyed by the header row; _row is the row's number in the sheet.
function rows_(name) {
  var v = sheet_(name).getDataRange().getValues(); var h = v[0], out = [];
  for (var i = 1; i < v.length; i++) { var o = {}; for (var j = 0; j < h.length; j++) o[h[j]] = v[i][j]; o._row = i + 1; out.push(o); }
  return out;
}
function rowsTail_(name, n) {                                                         // the last n rows only: submit_ runs under the lock, and a full read grows with the study
  var sh = sheet_(name), last = sh.getLastRow(), cols = sh.getLastColumn(), out = [];
  if (last < 2) return out;
  var h = sh.getRange(1, 1, 1, cols).getValues()[0], from = Math.max(2, last - n + 1), v = sh.getRange(from, 1, last - from + 1, cols).getValues();
  for (var i = 0; i < v.length; i++) { var o = {}; for (var j = 0; j < h.length; j++) o[h[j]] = v[i][j]; out.push(o); }
  return out;
}
// Appends objects to a tab in one write, in the tab's column order; missing fields become empty cells.
function append_(name, objs) {
  if (!objs.length) return;
  var h = TABS[name], data = objs.map(function (o) { return h.map(function (k) { return (o[k] === undefined || o[k] === null) ? "" : o[k]; }); });
  var sh = sheet_(name); sh.getRange(sh.getLastRow() + 1, 1, data.length, h.length).setValues(data);
}
// Minutes after which an assignment that was not completed stops holding its slot.
function timeoutMin_(task) {
  var p = props_().getProperty(task === "pairwise" ? "TIMEOUT_MIN_PAIRWISE" : "TIMEOUT_MIN_CALIBRATION");
  return p ? Number(p) : (task === "pairwise" ? 45 : 60);
}

// ---------------- one-off setup
// Creates the tabs and loads the session lists (index.json, written by scripts/02_generate_batches.py) into "batches".
function setup() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  Object.keys(TABS).forEach(function (name) {
    var sh = ss.getSheetByName(name) || ss.insertSheet(name);
    if (sh.getRange && sh.getMaxRows) sh.getRange(1, 1, sh.getMaxRows(), Math.max(sh.getMaxColumns(), TABS[name].length)).setNumberFormat("@");  // plain text: IDs never become numbers
    if (sh.getLastRow() === 0) sh.appendRow(TABS[name]);
  });
  var base = props_().getProperty("S3_BASE"); if (!base) throw new Error("set the S3_BASE script property first");
  var have = {}; rows_("batches").forEach(function (r) { have[r.task + "|" + r.batch_id] = true; });
  ["batches_pairwise", "batches_calibration"].forEach(function (dir) {
    var res = UrlFetchApp.fetch(base + dir + "/index.json", {muteHttpExceptions: true});
    if (res.getResponseCode() !== 200) {                                     // rating stage: no pairwise session lists on S3
      if (dir === "batches_pairwise") return;
      throw new Error("cannot read " + dir + "/index.json: finish the upload first");
    }
    var idx = JSON.parse(res.getContentText());
    append_("batches", idx.batches.filter(function (b) { return !have[idx.task + "|" + b]; })
                                  .map(function (b) { return {task: idx.task, batch_id: b, raters: idx.raters}; }));
  });
}

// ---------------- GET: ping, export, assign
function doGet(e) {
  var p = e.parameter || {};
  try {
    if (p.action === "ping") return json_({ok: true});
    if (p.action === "export") {
      if (!p.key || p.key !== props_().getProperty("EXPORT_KEY")) return json_({ok: false, error: "forbidden"});
      var out = {}; Object.keys(TABS).forEach(function (n) { out[n] = sheet_(n).getDataRange().getValues(); });
      return json_({ok: true, tabs: out, exported: new Date().toISOString()});
    }
    if (p.action === "assign") return json_(assign_(p));
    return json_({ok: false, error: "unknown action"});
  } catch (err) { return json_({ok: false, error: String(err)}); }
}

// Chooses the participant's session. The script lock makes the read-then-append atomic, so two participants who
// arrive together cannot take the same slot.
function assign_(p) {
  var task = p.task, pid = String(p.pid || ""), pilot = p.pilot === "1" ? 1 : 0;
  if (!pid || (task !== "pairwise" && task !== "calibration")) return {ok: false, error: "missing pid or task"};
  var lock = LockService.getScriptLock(); lock.waitLock(25000);
  try {
    var now = new Date(), tmo = timeoutMin_(task) * 60000;
    var A = rows_("assignments").filter(function (r) { return r.task === task; });
    var mine = A.filter(function (r) { return String(r.pid) === pid && r.status !== "excluded"; });
    if (mine.length) {
      var last = mine[mine.length - 1];
      if (last.status === "complete") return {ok: false, status: "already_complete"};
      sheet_("assignments").getRange(last._row, 1).setValue(now.toISOString());   // a reload keeps the same session
      return {ok: true, batch_id: String(last.batch_id), resumed: true};
    }
    var B = rows_("batches").filter(function (r) { return r.task === task; });
    // Per session, among the non-pilot participants: fill = slots held (completed, or assigned and not timed out) and
    // done = completed; pfill = pilot participants.
    var fill = {}, pfill = {}, done = {};
    B.forEach(function (b) { fill[b.batch_id] = 0; pfill[b.batch_id] = 0; done[b.batch_id] = 0; });
    A.forEach(function (r) {
      if (!(r.batch_id in fill)) return;
      if (Number(r.pilot) === 1) { pfill[r.batch_id]++; return; }
      if (r.status === "complete") done[r.batch_id]++;
      if (r.status === "complete" || (r.status === "assigned" && now - new Date(r.timestamp) < tmo)) fill[r.batch_id]++;
    });
    // Main study: the fullest session that still has a free slot, so that sessions complete one after another.
    // Pilot: the least-filled session, whatever its number of raters.
    var best = null;
    B.forEach(function (b) {
      var f = pilot ? pfill[b.batch_id] : fill[b.batch_id];
      if (!pilot && f >= Number(b.raters)) return;
      if (best === null || (pilot ? f < best.f : f > best.f)) best = {id: String(b.batch_id), f: f};
    });
    // A Prolific participant is never turned away. With every slot held, the session with the fewest completed raters (then
    // the fewest in progress) gets an extra rater; QC keeps the earliest clean raters, so an extra one is spare data, not harm.
    if (best === null) B.forEach(function (b) {
      var d = done[b.batch_id], f = fill[b.batch_id];
      if (best === null || d < best.d || (d === best.d && f < best.f)) best = {id: String(b.batch_id), d: d, f: f};
    });
    if (best === null) return {ok: false, status: "full"};
    append_("assignments", [{timestamp: now.toISOString(), pid: pid, study_id: p.study || "", session_id: p.session || "", task: task,
                             batch_id: best.id, status: "assigned", pilot: pilot}]);
    return {ok: true, batch_id: best.id};
  } finally { lock.releaseLock(); }
}

// ---------------- POST: save, submit, exclude
function doPost(e) {
  try {
    var b = JSON.parse(e.postData.contents);
    if (b.action === "save") return json_(save_(b));
    if (b.action === "submit") return json_(submit_(b));
    if (b.action === "exclude") return json_(exclude_(b));
    return json_({ok: false, error: "unknown action"});
  } catch (err) { return json_({ok: false, error: String(err)}); }
}

// Which tab a row belongs to, and the key that identifies one answer within a participant's session: the item of a
// pair, the image of a rating, or the grid of a top-three choice.
function respTab_(task) { return task === "pairwise" ? "responses_pairwise" : "responses_calibration"; }
function isPick_(r) { return r.pick1 !== undefined; }                                 // the top-three choices of a calibration session
function tabOf_(task, r) { return task === "calibration" && isPick_(r) ? "responses_selection" : respTab_(task); }
function key_(task, r) { return task === "pairwise" ? String(r.item_id) : isPick_(r) ? "pick|" + String(r.grid_id) : String(r.grid_id) + "|" + String(r.img); }
function appendSplit_(b, rows) {
  var by = {}; rows.forEach(function (r) { var t = tabOf_(b.task, r); (by[t] = by[t] || []).push(r); });
  Object.keys(by).forEach(function (t) { append_(t, stamp_(b, by[t])); });
}

// Adds the server's time and the participant's ids to every row.
function stamp_(b, rows) {
  var now = new Date().toISOString();
  return rows.map(function (r) {
    var o = {}; for (var k in r) o[k] = r[k];
    o.timestamp = now; o.pid = b.pid; o.study_id = b.study || ""; o.session_id = b.session || ""; o.batch_id = b.batch_id; return o;
  });
}

// Backup save during the session. Rows may arrive more than once; quality control keeps the last row per answer.
function save_(b) {
  var lock = LockService.getScriptLock(); lock.waitLock(25000);
  try { appendSplit_(b, b.rows || []); return {ok: true, n: (b.rows || []).length}; }
  finally { lock.releaseLock(); }
}

// Final submission: makes sure that every answer is stored, records the submission and marks the assignment complete.
// The page returns the participant to Prolific only after {ok: true}.
function submit_(b) {
  var own = {}; (b.rows || []).forEach(function (r) { own[key_(b.task, r)] = true; });
  var lock = LockService.getScriptLock(); lock.waitLock(25000);
  try {
    var have = {};
    if (Object.keys(own).length >= Number(b.n_items)) {                                // the page sent every answer, so store them all without reading the
      appendSplit_(b, b.rows); have = own;                                             // tabs (the slow part under load); rows saved earlier are duplicates, and QC keeps the last
    } else {                                                                           // the page lacks some answers (e.g. a new browser): count what the Sheet already has
      (b.task === "pairwise" ? ["responses_pairwise"] : ["responses_calibration", "responses_selection"]).forEach(function (tab) {
        rowsTail_(tab, 3000).forEach(function (r) {                                   // older rows are stored again if not seen here; QC keeps the last, so that is harmless
          if (String(r.pid) !== String(b.pid) || String(r.batch_id) !== String(b.batch_id)) return;
          have[key_(b.task, tab === "responses_selection" ? {pick1: 1, grid_id: r.grid_id} : r)] = true;
        });
      });
      var missing = (b.rows || []).filter(function (r) { return !have[key_(b.task, r)]; });
      appendSplit_(b, missing); missing.forEach(function (r) { have[key_(b.task, r)] = true; });
    }
    var n = Object.keys(have).length;                                                  // calibration: every rated image and every set's choices
    if (n < Number(b.n_items)) return {ok: false, error: "incomplete", stored: n, expected: Number(b.n_items)};
    append_("submissions", [{timestamp: new Date().toISOString(), pid: b.pid, study_id: b.study || "", session_id: b.session || "", task: b.task,
                             batch_id: b.batch_id, n_items: n, duration_sec: b.duration_sec, screen_w: b.screen_w, screen_h: b.screen_h,
                             user_agent: b.user_agent, pilot: b.pilot ? 1 : 0}]);
    var A = rows_("assignments"), sh = sheet_("assignments"), col = TABS.assignments.indexOf("status") + 1;
    for (var i = A.length - 1; i >= 0; i--) {
      if (String(A[i].pid) === String(b.pid) && A[i].task === b.task && String(A[i].batch_id) === String(b.batch_id)) { sh.getRange(A[i]._row, col).setValue("complete"); break; }
    }
    return {ok: true, stored: n};
  } finally { lock.releaseLock(); }
}

// Marks assignments as excluded (sent by scripts/03_qc_and_payment.py --push), which frees their slots for new participants.
function exclude_(b) {
  if (!b.key || b.key !== props_().getProperty("EXPORT_KEY")) return {ok: false, error: "forbidden"};
  var lock = LockService.getScriptLock(); lock.waitLock(25000);
  try {
    var A = rows_("assignments"), sh = sheet_("assignments"), col = TABS.assignments.indexOf("status") + 1, n = 0;
    var want = {}; (b.pids || []).forEach(function (x) { want[String(x.pid) + "|" + x.task] = true; });
    A.forEach(function (r) { if (want[String(r.pid) + "|" + r.task] && r.status !== "excluded") { sh.getRange(r._row, col).setValue("excluded"); n++; } });
    return {ok: true, excluded: n};
  } finally { lock.releaseLock(); }
}
