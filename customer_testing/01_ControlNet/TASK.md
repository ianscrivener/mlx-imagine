# Task: swap a face into a photo

You test an image server named mlx-imagine. You reach the server through MCP tools.
Do the task below. Then fill in `FEEDBACK.md`.

## Rules

- Use only the MCP tools of the server and public web pages. Do not read the source code of the server.
- Do not ask a human for help. If you cannot continue, stop. Write the reason in `FEEDBACK.md`.
- Keep a short log of each step. Include your mistakes and your guesses. You need the log for `FEEDBACK.md`.
- Stop after 40 tool calls, even if the task is not done.
- Use only the two photos in this package. Do not share the result image.

## Input

The folder `images/` has two photos:

- `face_closeup`: a close-up of a face. This is the face to put in.
- `target_photo`: the other photo. The face in this photo gets replaced.

The file type is png, jpg or webp.

## Resource

The Hugging Face repository `Alissonerdx/BFS-Best-Face-Swap` holds LoRA files for face, head and body swap.
Each file fits one base model. The repository has a guide for each model family.
Read the guide for Qwen-Image 2.1.

## Goal

1. Use the base model Qwen-Image 2.1. Choose the BFS LoRA file that fits it.
2. Make an image of the target photo with the face from `face_closeup`.
3. Save the result image. Give its path.

The result is good when two things are true. The face is the face from `face_closeup`. The rest of the photo stays like `target_photo`.

## When you finish

Write these in your final answer:

1. The path of the result image, or the word "none".
2. A short summary of what you did. Use 10 lines or fewer.
3. The completed `FEEDBACK.md`.
