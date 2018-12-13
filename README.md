# bw2color
Project for the course Artificial Intelligence in Practice.
Use cGAN to colorize Anime images.
Reference: https://arxiv.org/abs/1611.07004

## tools

- dataset spider

    download [wall.alphacoders.com](https://wall.alphacoders.com) thumb image

    ```bash
    cd bw2color
    python tools/wallpaper.py
    ```

- image preprocess

    crop downloaded image to 256x256 and convert to grey

    ```bash
    cd bw2color
    python tools/preprocess.py
    ```