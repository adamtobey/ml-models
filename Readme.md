# What is this project?

It's a learning exercise designed to practice using numpy and bokeh as well as to get a visual intuition for what a few common machine learning models do and how their parameters influence that behavior. All the models are implemented from scratch using numpy with the guiding principle to keep the implementation at the level of linear algebra, or in other words, *no `for` loops allowed!*

# View it in action!

The site is hosted on Heroku at https://calm-cliffs-70784.herokuapp.com/. It's currently hosted on the free project tier, meaning that the first visit may take a moment to load, and the plots may not update in realtime.

For a faster experience, check out how to get it set up locally below.

# Getting set up locally

In main project directory:
1. `pip install pipenv` if pipenv is not already installed
2. `pipenv install`
3. `pipenv run python main.py` to start the server on `$PORT`, defaulting to 8000
4. Navigate to `localhost:8000` in a web browser
