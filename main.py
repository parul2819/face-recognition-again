from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.config import PICS_DIR, UI_DIR
from api.controllers import (
    admin_controller,
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

# Serve photo files and the UI (must stay after the routers above)
app.mount("/pics", StaticFiles(directory=str(PICS_DIR)), name="pics")
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
