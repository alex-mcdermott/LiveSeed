# Crowd study: instructions and rubric shown to participants

Shown on the page before rating (verbatim), with the two tables kept on screen beside every image.

You will first see images one at a time, each with the text prompt that produced it. Give every image two scores from 1 to 5. Afterwards, you will choose your favourite images from a few sets.
The two scores are independent. A beautiful image can score 5 for quality and 1 for alignment if it shows the wrong thing. An image can score 5 for alignment and 2 for quality if everything requested is there but it looks mangled. Score each image on its own, using the descriptions below.

## Prompt alignment: did it follow the prompt?

| Score | Description |
|---|---|
| 5 | All subjects, attributes, counts and spatial relations are exactly as requested. |
| 4 | The core is correct, but one minor detail is missing or slightly wrong. |
| 3 | The main subject is present, but there is a serious error: a subject is missing, the count is wrong, or attributes are swapped. |
| 2 | Only loosely related: it has the general theme or one requested object, but ignores the main structure. |
| 1 | Nothing to do with the prompt. |

## Image quality: ignore the prompt, how well made is it?

| Score | Description |
|---|---|
| 5 | Flawless: no AI artefacts, correct anatomy and perspective, coherent textures. |
| 4 | Good: very minor imperfections that you only notice on close inspection. |
| 3 | Acceptable: noticeable artefacts, but objects are recognisable and intact. |
| 2 | Poor: severe structural problems or heavy artefacts, such as mangled limbs or severe blur. |
| 1 | Unusable: objects fused into impossible shapes, or a corrupted image of unrecognisable forms or noise. |

## Worked examples

Shown to every participant before rating (images in `rubric_examples/`).

| Image | Prompt | Prompt alignment | Image quality |
|---|---|---|---|
| `ex1.jpg` | Two cooked hot dogs on a plate with parsley. | 5: Two hot dogs, on a plate, with parsley: everything requested is there. | 5: Clean, realistic and free of visible errors. |
| `ex2.jpg` | a airplane hidden by a bag | 3: There is an aeroplane and a bag, but the bag does not hide the aeroplane. | 4: Clear and coherent, apart from slightly mangled wings and engines. |
| `ex3.jpg` | a leather jacket and a glass vase | 2: The vase is there, but the jacket is missing, and flowers were not requested. | 5: A clean, convincing image. It is well made even though it misses the prompt. |
| `ex4.jpg` | A person is kissing someone they love. | 5: It shows exactly what was asked. | 2: The hand merges into the faces and the fingers are badly formed. |
| `ex5.jpg` | a giraffe on the left of a train | 4: The main request is met, but a second giraffe was not requested. | 3: Mostly intact, but the giraffe's legs are unnaturally long. |
| `ex6.jpg` | a bowl in front of a cow | 2: It has a bowl and part of a cow, but the cow is inside the bowl, not behind it. | 2: A cow's head with no body, and garbled text on its ears. |
| `ex7.jpg` | a horse on side of a bicycle | 4: Both are there and side by side, though they are tangled together. | 1: The horse's body and legs fuse into the bicycle, and the frame and wheels are a tangle. |
| `ex8.jpg` | a person hidden by a book | 2: The book is there, but no person can be recognised. | 1: A dark shrouded shape melts into the page, and the book's structure makes no sense. |
| `ex9.jpg` | a fire hydrant that has dirty, brown water inside | 1: There is no fire hydrant, just a hole in the pavement. | 4: A clear, detailed image, even though it shows the wrong thing. |

Each session also contained two check images (scrambled copies of an image of the same prompt, whose image quality must be rated 1-2); they are not part of the release.
