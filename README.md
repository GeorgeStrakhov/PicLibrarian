# PICLIBRARIAN

_takes a library of images and makes them semantically searchable_

## How to use

0. save your openai and replicate keys in .env (see env.example)

1. put your images into `./images/`

2. `pip install -r requirements.txt`

3. `python piclibrarian.py` - this runs a script that:
    - goes through every image in the folder and renames it to URL friendly
    - creates a subfolder with resized smaller images (because we don't want to overload replicate and for captioning it doesn't matter that much)
    - for every image in the resized subfolder: runs clip interrogator to create an image description
    - passes the output of clip interrogator to GPT-4 to humanize it a bit
    - builds an embedding of the GPT-4 output
    - adds the filename, caption and embedding into a simple csv
    - once all images are captioned and embeddings built - csv is ready
    - the folder with resized smaller images is deleted (mom, you always told me to clean up after myself!)

4. `python picsearch.py -q "my query here"` - this runs a script that:
    - checks that the .csv from the previous step is done and finished
    - builds an embedding of your query
    - does vector similarity search
    - finds the closest match
    - returns the filename of the image that is the closest match to your search
