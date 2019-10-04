# Copyprism

## from photo to product copy

[Copyprism](http://intrepidanalytics.info/copyprism) is a web application for generating draft product copy from a picture of the product.

Under the hood, copyprism uses a combination of [Google Vision](https://cloud.google.com/vision/) and the [GPT-2 model](https://openai.com/blog/better-language-models/) to generate text descriptions. I fine-tuned the GPT-2 model on the [Ikea product catalogue](https://www.ikea.com/us/en/), meaning that product descriptions will be in the Ikea catalogue style. The model even knows to use Ikea product family names.

I used [Google Colab](https://colab.research.google.com/) for fitting/fine-tuning the GPT-2 model.

The slides from my demonstration are available [here](http://bitly.com/copyprism).

This project was created as part of the Insight Data Science program (Seattle 19C).



# relevant directory structure

- R: miscellaneously plotting script that is not actually used
- flask: partially self-contained flask web app (missing authentication information for Google Vision)
- misc: jupyter notebooks generated during this project. 
- src: python scripts for scraping Ikea website, splitting train/test data, fitting early draft rNN/LSTM model, validation on testing data. also includes one jupyter notebook for fine-tuning the GPT-2 model using Google Colab for free GPU time.
