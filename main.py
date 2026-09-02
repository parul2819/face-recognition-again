from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import auth
from api.config import PICS_DIR, UI_DIR
from api.controllers import (
    admin_controller,
    download_controller,
    identify_controller,
    image_controller,
    persons_controller,
    search_controller,
)
from api.db import close_pool, init_pool

app = FastAPI(title="Face Recognition Search API")


@app.on_event("startup")
async def startup():
    await init_pool()


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


app.include_router(auth.router)
app.include_router(persons_controller.router)
app.include_router(search_controller.router)
app.include_router(image_controller.router)
app.include_router(identify_controller.router)
app.include_router(admin_controller.router)
app.include_router(download_controller.router)


# admin.html itself is served unauthenticated -- it's just markup/JS with no
# sensitive data, and gates itself client-side via GET /admin/session,
# showing a login form until that succeeds. All the actual admin data comes
# from /admin/* routes, which do require a valid session (see api/auth.py).
# Served via an explicit route instead of the general UI static mount below
# so it can keep the no-store header (browsers were caching this page
# across deploys, silently hiding admin UI fixes/updates until a hard
# refresh).
@app.get("/admin.html")
async def admin_page():
    return FileResponse(str(UI_DIR / "admin.html"), headers={"Cache-Control": "no-store"})


# Serve photo files and the UI (must stay after the routers/admin.html
# route above -- FastAPI matches routes in registration order, and this
# StaticFiles mount at "/" would otherwise catch "/admin.html" first)
app.mount("/pics", StaticFiles(directory=str(PICS_DIR)), name="pics")
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
