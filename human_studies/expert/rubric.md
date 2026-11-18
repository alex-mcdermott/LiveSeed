# Expert study: rubric

Three annotators rated each of the 800 images (400 SDXL, 400 SD3.5-Medium) on the two scales below, one image at a time with its prompt.

Your Task: You will be shown a text prompt and an AI-generated image. You must evaluate the image on two independent scales: Prompt Alignment (Did it follow the instructions?) and Image Quality (Does it look like a good, structurally sound image?).
Please grade these strictly independently. An image can be a beautiful, flawless 5/5 for Quality, but a 1/5 for Alignment if it drew the wrong thing or lacks global coherence.
Conversely, an image can be a 4/5 for Alignment (everything requested is present), but a 2/5 for Quality if the faces are mangled and the image is smeary.

## Prompt Alignment (Semantic Adherence)

This scale measures how accurately the model bound the specific text concepts (counts, colors, spatial layouts, and objects) to the final image.

| Score | Rating | Definition | Example (prompt: "Two blue cats sitting on a red car") |
|---|---|---|---|
| 5 | Perfect Adherence | All core subjects, attributes, quantities, and spatial relationships are present exactly as requested. No missing elements or hallucinations. | Two cats, both are blue, and both are clearly sitting on top of a red car. |
| 4 | Minor Flaws | The core subjects and actions are correct, but one minor detail is missing or slightly misrepresented (e.g., wrong background, slight color bleed). | Two blue cats are next to a red car (spatial error), OR one cat is blue and the other is grey. |
| 3 | Major Flaw (Partial) | The main subject is present, but a severe compositional error occurred. A subject is missing, the count is totally wrong, or attributes are severely swapped. | There is only one blue cat on a red car (counting error), OR there are two red cats on a blue car (attribute swap). |
| 2 | Severe Failure | The image loosely relates to the general theme or contains one of the requested objects, but entirely ignores the primary compositional structure. | The image just shows a normal red car. The cats are entirely missing. |
| 1 | Complete Disconnect | The image has absolutely nothing to do with the prompt. Total generative failure. | The image shows a landscape of mountains, or random abstract noise. |

## Image Quality (Aesthetic & Structural Coherence)

This scale measures the visual fidelity, structural soundness, and lack of generative artifacts, completely ignoring whether it matched the text prompt.

| Score | Rating | Definition | Visual indicators |
|---|---|---|---|
| 5 | Flawless (could be a camera) | Visually stunning, structurally coherent, and highly realistic (or perfectly stylized). No AI artifacts, correct anatomy, and proper perspective. | Clean lines, physically accurate lighting, perfect faces/hands, coherent textures. |
| 4 | Good | High quality overall, but contains very minor, easily ignorable generative imperfections upon close inspection. Perhaps objects are infront/behind each other incorrectly. | Slightly asymmetrical eyes, a strange background shadow, or a slightly weird finger. |
| 3 | Acceptable | Noticeable AI artifacts that detract from the image, but the core structure of the objects remains generally recognizable and intact. | Warped background elements, slightly melted textures, or minor anatomical impossibilities. |
| 2 | Poor | Severe structural issues and heavy artifacting that make the image highly unnatural or confusing to look at. | Mangled limbs, disconnected torsos, "deep-fried" contrast (CFG burn), or severe blurriness. |
| 1 | Unusable | Completely corrupted generation. The image is a catastrophic failure of visual structure. | Unrecognizable amalgamations of flesh/objects, total noise, or pure static. |
