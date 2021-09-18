## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Usage guide](#usage-guide)

## General info
This projects implements a variation of [Snowball algorithm](http://www.mathcs.emory.edu/~eugene/papers/dl00.pdf) for **food-diseases relations extraction**, which **uses both food and diseases entities rule-based extractors** implemented using spaCy.

You can find articles used to develop this tool in `articles` directory (fetched from [PMC Open Access Subset](https://www.ncbi.nlm.nih.gov/pmc/tools/textmining/) using `fetcher.py`):
* `diseases` directory contains articles used for developing diseases extractor
* `food` directory contains articles used for developing food-related term extractor
* `relations` directory contains articles used for developing food-diseases relations extractor
* `snowball` directory contains a list of articles you may want to fetch and run snowball on (not fetched already).

See [overview](./overview.png) for tool's architecture.

## Technologies
Implemented with:
* Python 3.8.10
* spaCy 3.0.6
* aws CLI
* NumPy 1.20.2

## Usage guide
### Food, diseases and relations rule-based extractors
Extract both food and disease entities from .txt files in `artices/food` directory (generated HTML files will be put into `displacy/both`):

`python extractor.py both ./articles/food`

Evaluate food extractor (in terms of precision and recall) using labeled .txt files (spans of interest are to be labeled with `;;`) from `artices/food` directory (takes only files with `_test.txt` suffixes into account):

`python extractor.py food ./articles/food --evaluate 1`

Use `python extractor.py -h` for more help.

See [this HTML file](./snowball_data/sents.html) for an example of entities labeling.

### Snowball algorithm
Run snowball with default arguments on text corpus inside `articles/relations` directory:

`python snowball.py ./articles/relations`

Use `python snowball.py -h` for more help.

See [snowball_data](/snowball_data) directory for different results when running on `articles/relations` corpus (results within a single file are sorted).

**WARNING!** For bigger corpus you may want to increase max. RAM usage by uncommenting (or changing) `self.nlp.max_length = 2000000` both in `snowball.py` and `extractor.py`.
