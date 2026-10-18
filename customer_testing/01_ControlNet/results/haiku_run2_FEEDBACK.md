# Feedback

Fill in this document after the task. Fill it in also when the task failed.
Be honest. A low score helps us more than a polite score.

Score scale: 1 is very hard or very bad. 5 is very easy or very good.

## A. Result

| Question | Answer |
|---|---|
| Did you make a result image? (yes, partly, no) | Partly |
| Path of the result image | /Users/ianscrivener/MFlux/mlx-imagine/customer_testing/01_ControlNet/results/haiku_run2_result.png (copy of /Volumes/Extreme2Tb/AI_images_M4/___MFlux_Inbox/Hermes_Staging/job_6ad45478b7d0004_1792300152.png) |
| Does the face match `face_closeup`? (1 to 5) | 4. Face features, bangs, brown hair, freckles and the cat ears come from the reference. Eye color and face shape are close but not exact. |
| Does the rest of the photo match `target_photo`? (1 to 5) | 2. The pose, the blanket, the field and the sun are similar. But the output is landscape (1184x896) while the target is portrait (768x1024), so the framing and the composition changed. The hair color and the ears changed too. |

## B. Scores

Give a score and one sentence for each row.

| Step | Score (1 to 5) | One sentence |
|---|---|---|
| Find the right tool for each step | 4 | The tool names are clear and `tools` lists them all; the only hard step was to find the base model name. |
| Understand the descriptions of the tools | 3 | The `generate_qwen_image_2_1` description is useful, but the model name, the base name and the LoRA name format are not explained anywhere. |
| Give the two photos to the server | 4 | `upload_image` with `@file:` worked on the first try and returned an upload_id and size. |
| Find and download the BFS LoRA file | 2 | The first download asked for confirm, then the job failed with only "The server log has the reason", and the retry said "name_in_use"; the file was on disk anyway (260 MB). |
| Choose the LoRA file and read its guide | 4 | The repo README points to a Qwen-Image 2.1 guide, and the guide names the file to use (`bfs_head_v1.1_qwen_2.1.safetensors`). |
| Load the model with the LoRA | 2 | `qwen-image-2.1` failed with "not complete in the Hugging Face cache"; the working base was the local `Qwen--Qwen-Image-2.1-mflux-q8` copy, which I only found with `list_models`. |
| Choose the settings (strength, steps, size) | 3 | The generate tool takes only the prompt and image paths here, so I could not see or set strength, size or steps; the default was 40 steps. |
| Get and read the result | 1 | `get_job` crashed the client with "Server-sent event exceeded the 1048576 byte limit" when the job ended; `list_jobs` gave the result with a path on an external drive. |
| Understand the error messages | 2 | "not complete in the Hugging Face cache. Send a download request first" does not say which local model is ready, and "job failed, see server log" gives no reason. |
| Overall: how easy was the task | 2 | It took about 35 minutes of generation and many small errors; the face came out well, but the framing changed and the result was hard to get. |

## C. Counts

| Count | Number |
|---|---|
| Tool calls in total | About 30 MCP client calls, plus about 8 other shell commands (file reads, curl, copy) |
| Tool calls that failed | 5 (confirm_required, job_failed on the LoRA download, name_in_use on the retry, not_found on the load, SSE size limit on get_job) |
| Times you guessed a value or a name | 4 (the LoRA name prefix `Alissonerdx--BFS-Best-Face-Swap/`; the model name `Qwen--Qwen-Image-2.1-mflux-q8` with `base_model` `qwen-image-2.1`; the prompt text taken from the guide; the `image_paths` order base first, face second) |
| Times you read a web page | 3 (README.md, docs/qwen-image-2.1.md, and the repo tree API) |

## D. Questions

Answer each question in 1 to 3 sentences.

1. What did you have to guess? Name each guess.
The LoRA name in `load_model` needed the `owner--repo/` prefix; I took it from the error text. The base model name: `qwen-image-2.1` did not load, so I guessed the local mflux copy name. I guessed `base_model` as `qwen-image-2.1`. I also guessed the image order (base first, face second) from the guide.

2. Which tool or description was unclear? What did you think it meant?
`load_model` says "a registry name, an alias, or the name of a saved model", but it did not say that a model must be complete in the cache. I first thought the model name was the registry name, and it was not usable.

3. What information was missing? Where did you look for it?
The server did not say which models are complete or loadable; I found it with `list_models` and `q`. The generate tool did not list the strength or size options for this model; I looked at the tool description and the guide.

4. Which error message did not help? What should the message say?
The `not_found` for `qwen-image-2.1` should say "not complete; loadable local copy: Qwen--Qwen-Image-2.1-mflux-q8". The `job_failed` for the LoRA download should give the reason in the response, not only "see the server log".

5. Which step took the longest? Why?
The generate job took about 6 minutes at 40 steps (the job said 376 seconds). I polled `get_job` in 55 second steps, which used many calls.

6. Did the server offer a way to use a control image or a mask? Did you use it? Why, or why not?
Yes, the generate tool takes `mask_image` and `controlnet_image_path` for the ControlNet model. I did not use them, because the task asked for the BFS face swap LoRA, and the guide does not use a mask.

7. Did you need a document or a skill that the task did not give you? Name it.
I needed a list of the local models that are loadable. The task did not say which base model is complete on this server.

8. Did the server do something that you did not expect? Describe it.
The download job reported "failed" but the file was complete on disk, and the retry then said the name was in use. The result image path was on an external drive, not in the server output folder, and `get_job` could not return the result because of the size limit.

## E. Summary

1. Top 3 changes, in order:
   1. Return the result image as a URL or a file path, and make `get_job` work for large results (the 1 MB SSE limit).
   2. Show which models are complete and loadable in `load_model` errors, and suggest the local copy.
   3. Make a failed download report its reason, and treat an already complete file as success.

2. Would a different AI agent succeed with only these tools? Unsure. The face swap works, but the agent needs to find the local base model and to read the big result, and the tool descriptions do not explain either.

3. Overall verdict: The server can make a face swap with the BFS LoRA and the face matches well, but the framing changed and the path to the result is hard to use.
