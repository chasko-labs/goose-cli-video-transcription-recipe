# goose project shape (lives outside this repo)

this repo holds the recipes. the goose project shape — the hints file and `.goose/` discovery dir that make sessions in `~/video-transcripts/` route to those recipes — lives in the data dir, not here. this doc records the shape so future contributors know where to look

## where things live

- recipes (this repo): `~/code/projects/goose-cli-video-transcription-recipe/recipes/*.yaml`
- project hints (data dir): `~/video-transcripts/goosehints.md` + `~/video-transcripts/.goosehints` → goosehints.md
- recipe discovery (data dir): `~/video-transcripts/.goose/recipes` → symlink into this repo's `recipes/`
- config placeholder (data dir): `~/video-transcripts/.goose/config.yaml` — tbd, goose-cli does not yet document a per-project config.yaml schema

## why split

the data dir is where bryan opens sessions; the recipes are versioned source in this repo. symlinks keep the recipe yamls in one place while making them discoverable via goose's documented project path `.goose/recipes/`

## verification notes (issue #24)

- goosehints.md sections: what-this-project-is, recipes, knowledge-flow, agent-personas-using-this-corpus
- word count target: under 400 (context-efficiency pattern — do not duplicate derivable info like dir listings, git remote, recipe contents)
- in-session smoke test deferred: goose binary not installed on rocm-aibox; route-to-recipe check runs at next install
- the four recipes in the hints file: transcribe, tutor, tidy, log. tidy may still be in flight per #21; log targets `~/code/heraldstack/gandergoosecli/` inbox-capture which is currently sparse (only `goose-fallback.sh` present) — both are flagged tbd in the hints file
