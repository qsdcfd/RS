{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "liblecture",
      "provenance": [],
      "authorship_tag": "ABX9TyORgM56BVb5CQ2sd/AQhm+U"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AkJxRCuRsDHZ"
      },
      "outputs": [],
      "source": [
        "from IPython.display import display, HTML\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import math\n",
        "\n",
        "\n",
        "\n",
        "def displayMovies(movies, movieIds, ratings = []):\n",
        "\n",
        "    html = \"\"\n",
        "\n",
        "    for i, movieId in enumerate(movieIds):\n",
        "        movie = movies[movies['movieId'] == movieId].iloc[0]\n",
        "\n",
        "        html += f\"\"\"\n",
        "            <div style=\"display:inline-block;min-width:150px;max-width:150px; vertical-align:top\">\n",
        "                <img src = '{movie.imgurl}' width=120> <br/>\n",
        "                <span>{movie.title}</span> <br/>\n",
        "                {f'<span>{ratings[i]}</span> <br/>' if len(ratings) > 0 else \"\"}\n",
        "                <ul>{\"\".join([f\"<li>{genre}</li>\" for genre in movie.genres.split('|')])}</ul>\n",
        "            </div>            \n",
        "        \"\"\"\n",
        "    \n",
        "    display(HTML(html))\n",
        "\n",
        "\n",
        "def getMAE(real, pred):\n",
        "     errors = real - pred\n",
        "     return errors.abs().mean()\n",
        "\n",
        "def getRMSE(real, pred):\n",
        "    errors = real - pred\n",
        "    return math.sqrt(errors.pow(2).mean())"
      ]
    }
  ]
}