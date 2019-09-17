# from image to seller's copy

## idea

given a picture of an object, produce a rough draft of the copy that would appear for that item on a seller's website (e.g. amazon, ikea, wayfair, ebay, etc.).

copy == item description

**input:** image file

**output:** natural language paragraph


## utility

writing copy costs time and money (in-house, freelance, etc)

editing is *easier* than writing.

writing object copy probably only dings those $1/hour things being offered on upwork.

means you can just get a native speaker for less time while paying them reasonably.

decrease writing budget overhead.


## implementation

### data

scrape *images* and *copy* from seller sites

- amazon
- ikea
- wayfair
- etsy (API)
- west elm (robots.txt says no)
- design within reach

**note:** images are used for validation/testing data


### train model

copy text options

- markov chain generator
  - not as sophisticated/fancy
  - possibly better at human readable text
  - purpose is to generate 
  - R or python
- recurrent neural network / LSTM 
  - e.g. keras
  - fancy/sophisticated
  - people love those words
  - R or python

then, given a few seed words, output a rough draft paragraph


### test model

- image into google vision (python)
- extract key terms
- use key terms to seed newly generated copy
- measure (dis)similarity between human written and computer generated
# copyprisim
