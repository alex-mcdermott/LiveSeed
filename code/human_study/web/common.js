// common.js: the page code shared by the crowd study's task pages (paper App. E): consent, the desktop check, session
// assignment, saving with retries and a local backup, and the confirmed submission.
// Loaded by calibration.html (the rating task) after config.js, which scripts/02_generate_batches.py writes and which
// defines window.STUDY_CFG. The backend that answers the requests made here is apps_script/Code.gs.
// The "pairwise" branches belong to a second task (two images side by side) that the study did not run.
"use strict";

const LS = (function () {
  const C = window.STUDY_CFG;
  const q = new URLSearchParams(window.location.search);
  // What Prolific appends to the study link: the participant, study and session ids. ?pilot=1 marks a pilot run.
  const P = {
    pid: q.get("PROLIFIC_PID") || "",
    study: q.get("STUDY_ID") || "",
    session: q.get("SESSION_ID") || "",
    pilot: q.get("pilot") === "1",
  };

  function $(id) { return document.getElementById(id); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  // Returns a shuffled copy (Fisher-Yates).
  function shuffle(a) { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

  // The task needs a large window: no phones or tablets, and at least 1000 pixels of width.
  function isDesktop() {
    const mobile = /Mobi|Android|iPhone|iPad|iPod|Tablet/i.test(navigator.userAgent);
    return !mobile && window.innerWidth >= 1000 && window.screen.width >= 1000;
  }

  // Replaces the page by a message; used when the study cannot start.
  function fail(msg) {
    document.body.innerHTML = '<div class="container"><h2>Sorry</h2><p>' + msg + '</p><p>Questions: ' + C.contactEmail + '</p></div>';
  }

  // GET with retries (session assignment)
  async function getJSON(url, tries) {
    let last;
    for (let t = 0; t < (tries || 5); t++) {
      try { const r = await fetch(url, {cache: "no-store"}); if (r.ok) return await r.json(); last = "HTTP " + r.status; }
      catch (e) { last = String(e); }
      await sleep(800 * Math.pow(2, t));
    }
    throw new Error(last);
  }

  // POST as text/plain (a "simple" request: no CORS preflight, which Apps Script does not answer)
  async function post(body, tries) {
    let last;
    for (let t = 0; t < (tries || 5); t++) {
      try {
        const r = await fetch(C.appsScriptUrl, {method: "POST", headers: {"Content-Type": "text/plain;charset=utf-8"}, body: JSON.stringify(body), redirect: "follow"});
        const j = await r.json(); if (j.ok) return j; last = j.error || JSON.stringify(j);
        // "incomplete" (the server holds fewer answers than expected) is returned at once: repeating the request cannot change it.
        if (j.error === "incomplete") return j;
      } catch (e) { last = String(e); }
      await sleep(800 * Math.pow(2, t));
    }
    return {ok: false, error: last};
  }

  // local backup: every answer is kept in localStorage until the server confirms the submission
  function storeKey(task) { return "liveseed_" + task + "_" + P.pid; }
  function loadStore(task) { try { return JSON.parse(localStorage.getItem(storeKey(task)) || "null"); } catch (e) { return null; } }
  function saveStore(task, st) { try { localStorage.setItem(storeKey(task), JSON.stringify(st)); } catch (e) {} }

  // Answers go to the server in batches, not one at a time, because every save holds the backend's single lock and a whole
  // study can run at once: in a load test, sixty raters at once saturated the lock at about one call a second. The queue
  // is sent once it holds 42 answers (all of Part 1), in requests of at most 25 rows, and otherwise only every ten minutes.
  // The final submission carries every answer itself, and each answer is already in localStorage, so these sends are only
  // a backup.
  const queue = [];
  let flushing = false;
  async function flush(task, batch) {
    if (flushing) return; flushing = true;
    try {
      while (queue.length) {
        const rows = queue.slice(0, 25);
        const r = await post({action: "save", task: task, pid: P.pid, study: P.study, session: P.session, batch_id: batch, rows: rows}, 3);
        if (!r.ok) break;
        queue.splice(0, rows.length);
      }
    } finally { flushing = false; }
  }
  function enqueue(task, batch, rows, now) { rows.forEach(function (r) { queue.push(r); }); if (now || queue.length >= 42) flush(task, batch); }
  setInterval(function () { if (LS._task && queue.length) flush(LS._task, LS._batch); }, 600000);

  // Final submission: sends every answer, waits for the server's confirmation and only then returns the participant to
  // Prolific with the completion code. Without confirmation the answers stay in localStorage, and the participant can try
  // again or use the fallback code.
  async function submit(task, batch, allRows, nItems, startedMs) {
    const box = $("finish-box"); box.style.display = "block";
    $("finish-msg").textContent = "Saving your answers, please wait…";
    queue.length = 0;                                                                  // the submission below carries every answer
    const body = {action: "submit", task: task, pid: P.pid, study: P.study, session: P.session, batch_id: batch, n_items: nItems,
                  duration_sec: Math.round((Date.now() - startedMs) / 1000), screen_w: window.screen.width, screen_h: window.screen.height,
                  user_agent: navigator.userAgent.slice(0, 200), pilot: P.pilot, rows: allRows};
    const r = await post(body, 6);
    if (r.ok) {
      try { localStorage.removeItem(storeKey(task)); } catch (e) {}
      $("finish-msg").textContent = "Thank you! Your answers are saved. Returning you to Prolific…";
      window.location.href = "https://app.prolific.com/submissions/complete?cc=" + encodeURIComponent(C.completion[task]);
      return;
    }
    $("finish-msg").innerHTML = "We could not confirm that your answers reached our server (" + (r.error || "network error") + "). " +
      "Your answers are kept in this browser. Please press <b>Try again</b>. If it still fails after a few tries, return to Prolific " +
      "and enter the completion code <b>" + C.fallback[task] + "</b>; you will be paid, and please message us so we can recover your answers.";
    const btn = $("retry-btn"); btn.style.display = "inline-block";
    btn.onclick = function () { btn.style.display = "none"; submit(task, batch, allRows, nItems, startedMs); };
  }

  // The information and consent text shown before the task. The bracketed placeholders replace identifying details.
  function consentHtml(task) {
    const what = task === "pairwise"
      ? "You\'ll see 22 pairs of AI-generated images, each with the text prompt that produced them, and answer three questions per pair: which image has fewer visual or structural errors, which you prefer overall, and which follows the prompt more closely."
      : "You\'ll see 42 images, one at a time, each with the text prompt that produced it. Please rate each one from 1 to 5 for (a) its objective image quality and (b) how well it follows the prompt. You\'ll then see 4 sets of 10 new images and choose your three favourites from each set.";
    return '' +
      '<h2>[Institution]: rating AI-generated images</h2>' +
      '<p><b>Purpose.</b> We\'re testing a method that helps AI text-to-image models produce better images, which people prefer. The research is part of a PhD at [institution] ([research centre]).</p>' +
      '<p><b>What you will do.</b> ' + what + ' It takes about ' + C.minutes[task] + ' minutes on a desktop or laptop computer; you\'ll be paid £' + C.pay[task].toFixed(2) + ' through Prolific.</p>' +
      '<p><b>Content.</b> The images are made by AI models from text prompts (for example everyday objects, animals, people and scenes). AI images can occasionally be distorted or unexpected. If anything makes you uncomfortable, you may stop at any time.</p>' +
      '<p><b>Your data.</b> We record your answers, how long you take, your screen size and your Prolific ID. The Prolific ID is used only to pay you and to check answer quality; we collect no name, email or other personal details. Answers are stored in a Google Sheet under the researcher\'s account. ' + C.retention + '. In the public dataset each participant is replaced by a random code that cannot be linked back to the Prolific ID.</p>' +
      '<p><b>Taking part is voluntary.</b> You may stop at any time by closing this page (return the study on Prolific). You can ask us to delete your answers until the dataset is anonymised and released, by messaging us on Prolific or by email with your Prolific ID.</p>' +
      '<p><b>Contact.</b> ' + C.contactEmail + '.</p>' +
      '<label class="consent-check"><input type="checkbox" id="consent-box"> I have read the information above, I am 18 or older, and I agree to take part. I understand that my anonymised ratings will be released publicly as a research dataset.</label>' +
      '<p><button class="btn" id="start-btn" disabled>Start</button></p>';
  }

  // shows the consent screen, then assigns a session and loads it; calls onReady(batchJson, storedState)
  async function start(task, onReady) {
    LS._task = task;
    if (!P.pid) { fail("This page must be opened from Prolific (the link carries your participant ID)."); return; }
    if (!isDesktop()) { fail("This study needs a desktop or laptop computer with a window at least 1000 pixels wide. Please return the study on Prolific, or reopen the link on a computer."); return; }
    $("consent-screen").innerHTML = consentHtml(task);
    $("consent-box").onchange = function () { $("start-btn").disabled = !this.checked; };
    $("start-btn").onclick = async function () {
      $("start-btn").disabled = true; $("start-btn").textContent = "Loading…";
      try {
        const aUrl = C.appsScriptUrl + "?action=assign&task=" + task + "&pid=" + encodeURIComponent(P.pid) + "&study=" + encodeURIComponent(P.study) +
                     "&session=" + encodeURIComponent(P.session) + (P.pilot ? "&pilot=1" : "");
        let a = await getJSON(aUrl);
        for (let t = 0; t < 6 && !a.ok && a.error; t++) { await sleep(1500 * Math.pow(2, t)); a = await getJSON(aUrl); }   // a busy server (lock timeout) is retried, not shown as an error
        if (!a.ok) { fail(a.status === "already_complete" ? "You have already completed this study, and that submission is saved. Thank you! Each person can take part only once, so please return this new attempt on Prolific: close this tab, open <a href=\"https://app.prolific.com/submissions\" target=\"_blank\" rel=\"noopener\">your Prolific submissions</a>, and choose <b>Return</b> for this study. Returning does not affect your Prolific record." : a.status === "full" ? "All places in this study are currently taken. Please return the study on Prolific; thank you for your interest." : "Could not start the study (" + (a.error || a.status) + ")."); return; }
        LS._batch = a.batch_id;
        const batch = await getJSON((task === "pairwise" ? "batches_pairwise/" : "batches_calibration/") + a.batch_id + ".json");
        let st = loadStore(task);
        if (!st || st.batch_id !== a.batch_id) st = {batch_id: a.batch_id, started: Date.now(), rows: {}};
        saveStore(task, st);
        $("consent-screen").style.display = "none";
        onReady(batch, st);
      } catch (e) { fail("Could not load the study (" + e.message + "). Please check your connection and reload the page."); }
    };
  }

  // image loading with retries; onBroken lets the participant report a broken image instead of being stuck
  function loadImg(img, src, onOk, onBroken) {
    let tries = 0;
    img.onload = function () { onOk(); };
    img.onerror = function () { tries++; if (tries <= 2) setTimeout(function () { img.src = src + (src.indexOf("?") < 0 ? "?" : "&") + "r=" + tries; }, 600 * tries); else onBroken(); };
    img.src = src;
  }
  // Fetches the next images in the background, so that they appear without delay.
  function preload(srcs) { srcs.forEach(function (s) { const i = new Image(); i.src = s; }); }

  function openZoom(src) { $("zoom-img").src = src; $("zoom-modal").style.display = "flex"; }

  return {C: C, P: P, $: $, shuffle: shuffle, start: start, enqueue: enqueue, submit: submit, saveStore: saveStore, loadImg: loadImg,
          preload: preload, openZoom: openZoom, _task: null, _batch: null};
})();
