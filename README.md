# Lecture Learner

Lecture Learner turns a transcript into Markdown study notes and course announcements. It also includes OpenAI audio transcription and Claude notes paths. The offline transcript mode needs only Julia.

## Quick start

Use Julia 1.13. From this folder:

```sh
julia --project=. LectureLearner.jl --offline --text sample_transcript.txt
```

This writes `sample_transcript_notes.md`. Check the notes against the transcript: the offline summary can miss details.

## Cloud setup

HTTP and JSON3 are listed in `Project.toml` and pinned in `Manifest.toml`. Install them with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Set API keys in your shell, never in project files. `OPENAI_API_KEY` is needed for audio transcription; `ANTHROPIC_API_KEY` is needed for the default Claude notes backend.

Audio transcription also needs [`ffmpeg`](https://ffmpeg.org) on your `PATH` (`brew install ffmpeg`). Recordings are transcoded to a compact format before upload, and lectures that still exceed Whisper's 25 MB limit are split into chunks automatically — so files of any length work.

```sh
julia --project=. LectureLearner.jl --text sample_transcript.txt
julia --project=. LectureLearner.jl lecture.m4a
```

If cloud notes fail, the program uses the offline summarizer. Audio still requires an OpenAI key. Results are written beside the input as `_notes.md`; audio also writes `_transcript.txt`. Existing output files with those names are replaced.

## Tests and files

```sh
julia --project=. test/runtests.jl
```

**Tested:** Offline transcript processing was manually tested with `sample_transcript.txt`, and the generated notes were reviewed. The automated regression tests passed, including mocked cloud-response fallback.

**Not tested:** Live cloud APIs and audio transcription were not tested.

`LectureLearner.jl` is the command-line program, `TextTools.jl` is the offline summarizer, and `docs/` holds the detailed current guide.

## Presentation note

Once the project is running, add its run steps and sample output to the slides so the team can present without a live run. If the team does run it live, skip that slide walkthrough. Ask Pao by email whether a live run is required.
