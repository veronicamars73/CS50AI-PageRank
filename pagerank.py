import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if corpus.get(page) is None or corpus.get(page) == {} or len(corpus[page])==0:
        probability_pages = 1/len(corpus)
        result_dict = {}
        for page_it in corpus.keys():
            result_dict[page_it] = probability_pages
    else:
        probability_pages_gen = 1/len(corpus)
        probability_pages_in = 1/len(corpus[page])
        result_dict = {}
        for page_it in corpus.keys():
            if page_it in corpus[page]:
                prob_page = (probability_pages_in*damping_factor)+(probability_pages_gen*(1-damping_factor))
            else:
                prob_page = (probability_pages_gen*(1-damping_factor))
            result_dict[page_it] = prob_page
    return result_dict
        

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result = {page: 0 for page in corpus}
    num_pages = len(result)

    for _ in range(n):
        # Start with a random page
        origin_page = random.choice(list(corpus.keys()))
        result[origin_page] += 1

        for _ in range(num_pages - 1):
            probabilities = transition_model(corpus, origin_page, damping_factor)
            page_surfed = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
            result[page_surfed] += 1
            origin_page = page_surfed

    # Normalize PageRank values
    result = {key: value / (n * num_pages) for key, value in result.items()}
    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    orgin_pagerank = {page_name: 1/len(corpus) for page_name in corpus}
    convergence = False

    while not convergence:
        iteration_pagerank = {} 

        for page in corpus:
            new_rank = (1 - damping_factor) / len(corpus)

            for potential_page, links in corpus.items():
                contribution = 0
                if page in links:
                    contribution = orgin_pagerank[potential_page] / len(links)
                new_rank += damping_factor * contribution
            
            iteration_pagerank[page] = new_rank
            
        convergence = True
        for page in orgin_pagerank.keys():
            if abs(iteration_pagerank[page] - orgin_pagerank[page]) > 0.001:
                convergence = False

        orgin_pagerank = iteration_pagerank

    return orgin_pagerank


if __name__ == "__main__":
    main()
