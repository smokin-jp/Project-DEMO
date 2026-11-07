"""FastAPI service. Run:  backend/.venv/Scripts/python -m uvicorn app.main:app --port 8000  (from ./backend)"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .clip_model import ClipScorer, Scorer
from .config import settings
from .extras import Extras, load_extras
from .judge import decode_jpeg, judge, judge_multi
from .schemas import JudgeRequest, MultiJudgeRequest, ThrowTypeRequest, ThrowTypeResult, Verdict
from .throw_clip import ThrowTypeModel

THROW_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "throw_clip.pt"

log = logging.getLogger("referee")


def create_app(scorer: Scorer | None = None, throw_model: ThrowTypeModel | None = None, extras: Extras | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if scorer is None:
            log.warning("loading CLIP %s on %s ...", settings.clip_model, settings.device)
            app.state.scorer = ClipScorer(settings.clip_model, settings.device).load()
            # warm-up so the first real punch is not slow
            from PIL import Image

            from .prompts import PROMPT_SETS

            # encode every prompt set once now (punch / kick / knee x each question): their text embeddings are cached,
            # so the first real strike of each type does not pay for them
            blank = Image.new("RGB", (224, 224))
            for sets in PROMPT_SETS.values():
                for prompt_set in sets.values():
                    app.state.scorer.classify([blank], prompt_set)
            log.warning("CLIP ready")
        else:
            app.state.scorer = scorer
        # the judo throw-type model is optional: it exists once training/03b_train_clip.py has been run
        app.state.throw_model = throw_model
        if throw_model is None and scorer is None and THROW_MODEL_PATH.exists():
            app.state.throw_model = ThrowTypeModel.load(str(THROW_MODEL_PATH))
            log.warning("throw-type model loaded: %s", THROW_MODEL_PATH.name)
        # the team's trained kick detector and punch-type model: loaded when their files exist (never in tests unless passed in)
        app.state.extras = extras if extras is not None else (load_extras(settings.device) if scorer is None else Extras())
        yield

    app = FastAPI(title="AI Referee: punch judge", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        s = app.state.scorer
        return {"ok": True, "model": getattr(s, "name", "?"), "device": settings.device, "loaded": getattr(s, "loaded", True),
                "throw_model": app.state.throw_model is not None, **app.state.extras.names}

    @app.post("/api/judge", response_model=Verdict)
    def judge_punch(req: JudgeRequest):
        try:
            return judge(req, app.state.scorer, extras=app.state.extras)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/api/judge_multi", response_model=Verdict)
    def judge_strike_multi(req: MultiJudgeRequest):
        """The same strike seen by several cameras (clips share one clock): one fused verdict."""
        try:
            return judge_multi(req, app.state.scorer, extras=app.state.extras)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/api/throw_type", response_model=ThrowTypeResult)
    def throw_type(req: ThrowTypeRequest):
        """Which judo throw was that? (ippon seoi nage / o goshi / osoto gari / uchi mata). Only ask when a throw-like fall happened."""
        tm = app.state.throw_model
        if tm is None:
            raise HTTPException(status_code=503, detail="no throw-type model installed (run training/03b_train_clip.py)")
        frames = sorted(req.frames, key=lambda f: f.t_ms)
        r = tm.predict([decode_jpeg(f.jpeg_b64) for f in frames], app.state.scorer)
        return ThrowTypeResult(**r, model_info={k: v for k, v in tm.meta.items() if k in ("val", "best_epoch", "clip_model")})

    return app


app = create_app()
