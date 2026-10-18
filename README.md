# Lecture Learner

Lecture Learner has two separate commands: local recording-to-text transcription, and the existing transcript-to-notes program. The notes program also retains its OpenAI audio and Claude notes paths.

## Quick start

Use Julia 1.13. From this folder:

```sh
julia --project=. LectureLearner.jl --offline --text sample_transcript.txt
```

This writes `sample_transcript_notes.md`. Check the notes against the transcript: the offline summary can miss details.

## Local transcription (recording to plain text)

On this Apple M5 Mac, FFmpeg 8.1 is installed. Install [whisper.cpp](https://formulae.brew.sh/formula/whisper.cpp) and download an [English `small.en` model](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md) yourself. The model is about 466 MiB; `models/` is ignored by Git.

```sh
brew install whisper.cpp
mkdir -p models
curl -fL --output models/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

For a 3-minute excerpt of a local MP4 (or other recording), run:

```sh
julia --project=. transcribe_local.jl "/full/path/lecture video.mp4" \
  --model models/ggml-small.en.bin --start 0 --duration 180 \
  --output transcripts/lecture_excerpt.txt
```

Omit `--start`, `--duration`, and `--output` for the full recording; the default output is `transcripts/<recording-name>_transcript.txt`. The command extracts temporary 16 kHz mono audio with FFmpeg, uses local `whisper-cli`, saves text without timestamps, prints elapsed processing time, and removes the temporary audio. It refuses to overwrite an existing transcript. Paths with spaces must be quoted. It does not upload recordings, download models, or run the notes program. On a Mac without FFmpeg, install it with `brew install ffmpeg`.

## Cloud setup

HTTP and JSON3 are listed in `Project.toml` and pinned in `Manifest.toml`. Install them with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Set API keys in your shell, never in project files. `OPENAI_API_KEY` is needed for audio transcription; `ANTHROPIC_API_KEY` is needed for the default Claude notes backend.

```sh
julia --project=. LectureLearner.jl --text sample_transcript.txt
julia --project=. LectureLearner.jl lecture.m4a
```

If cloud notes fail, the program uses the offline summarizer. Audio still requires an OpenAI key. Results are written beside the input as `_notes.md`; audio also writes `_transcript.txt`. Existing output files with those names are replaced.

## Tests and files

```sh
julia --project=. test/runtests.jl
```

**Tested:** Offline transcript-to-notes processing was manually tested with `sample_transcript.txt`, and the generated notes were reviewed. The automated regression suite passed, including local transcription tests with mock executables. A synthetic MP4 was converted by the installed FFmpeg to 16 kHz, mono, 16-bit PCM WAV; a mock `whisper-cli` completed the local command.

**Not tested:** Real speech recognition with whisper.cpp, the 2-3 minute lecture excerpt, a longer recording benchmark, live cloud APIs, and cloud audio transcription. `whisper-cli`, the model, and the lecture video were not available for those checks.

`transcribe_local.jl` is the standalone local transcription command. `LectureLearner.jl` and `TextTools.jl` remain the notes pipeline. `docs/` holds its detailed guide.

## Presentation note

Once the project is running, add its run steps and sample output to the slides so the team can present without a live run. If the team does run it live, skip that slide walkthrough. Ask Pao by email whether a live run is required.
