from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import *

import urllib.request
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

defaults.device = torch.device('cpu')


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

leaf_learner = load_learner(path = '.',file = 'leaf_classifier_2.pkl')



@app.route("/upload", methods=["POST"])
async def upload(request):
   data = await request.form()
   in_file = await (data["file"].read())
   out_file = open("photo.jpeg", "wb") # open for [w]riting as [b]inary
   out_file.write(in_file)
   out_file.close() 
   return predict_image()


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    url = request.query_params["url"]
    resp = requests.get(url, stream=True)
    # Open a local file with wb ( write binary ) permission.
    local_file = open('photo.jpeg', 'wb')
    # Set decode_content valule to True, otherwise the downloaded image file's size will be zero.
    resp.raw.decode_content = True
    # Copy the response stream raw data to local image file.
    shutil.copyfileobj(resp.raw, local_file)
 
    return predict_image()


def predict_image():
    img = open_image('photo.jpeg')
    pred,_,_ = leaf_learner.predict(img)
    return JSONResponse({
        "predictions": str(pred)
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
