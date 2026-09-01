from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import verify_admin_credentials
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


app.include_router(persons_controller.router)
app.include_router(search_controller.router)
app.include_router(image_controller.router)
app.include_router(identify_controller.router)
app.include_router(admin_controller.router)
app.include_router(download_controller.router)


# The admin page itself needs the same Basic Auth as the /admin/* API
# routes (same realm, so the browser's login prompt covers both after one
# entry) -- served via an explicit route instead of the general UI static
# mount below, which has no way to attach a dependency to just one file.
@app.get("/admin.html", dependencies=[Depends(verify_admin_credentials)])
async def admin_page():
    # no-store: browsers were caching this page across deploys, silently
    # hiding admin UI fixes/updates until a hard refresh.
    return FileResponse(str(UI_DIR / "admin.html"), headers={"Cache-Control": "no-store"})


# Serve photo files and the UI (must stay after the routers/admin.html
# route above -- FastAPI matches routes in registration order, and this
# StaticFiles mount at "/" would otherwise catch "/admin.html" first)
app.mount("/pics", StaticFiles(directory=str(PICS_DIR)), name="pics")
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
