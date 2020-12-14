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
    #raise NotImplementedError
    corpus_keys = list(corpus.keys())
    N_pages = len(corpus_keys)
    dist_proba = {}
    
    #Si la page entrée n'est pas dans le corpus:
    if page not in corpus_keys:
        return ("Page not found")
    
    pages = corpus[page]
    N_links = len(pages)
    
    #Si la page ne contient aucun link: équiprobabilité pour les pages
    if (len(pages)) == 0:
        dist_proba = {key:1/N_pages for key in corpus_keys}
        return dist_proba    
    #Sinon:
    else:
        for key in corpus_keys:
            #Si key est un link de la page:
            if key in pages:
                dist_proba[key] = damping_factor/N_links + (1 - damping_factor)/N_pages
            #Si key n'est pas un link de page:
            else:
                dist_proba[key] = (1 - damping_factor)/N_pages
        return dist_proba

#Vérifier que ma fonction transition_model fonctionne coorectly en prenant l'exemple de l'enoncé
"""     
corpus = {"1.html" : {"2.html", "3.html"}, "2.html" : {"3.html"},"3.html" : {"2.html"}}
test = transition_model(corpus , "2.html", 0.85)
print(test)
"""

def sample_pagerank(corpus, damping_factor, n):
    
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    #raise NotImplementedError
    #Initialiser la sortie
    pageranks = {}
    #Mettre les clés de corpus dans e liste
    corpus_keys = list(corpus.keys())
    #On initialise notre échantillon au hasrd
    echantillon = random.choice(corpus_keys)
    #Mettre la page dans une liste
    listes_pages = []
    listes_pages.append(echantillon)
    
    for i in range(1, n):
        #Appel de la fonction transition model pour tirer un échantillon
        dist_proba = transition_model(corpus, echantillon, damping_factor)
            
        probs = [dist_proba[key] for key in dist_proba.keys()]
        
        #On génère notre échantillon à partir de l'échantillon précédent sur la
        # base du modèle de transition de l'échantillon précédent avec la 
        #probabilité pi de chosir une page i.
        echantillon = random.choices(list(dist_proba.keys()), weights = probs).pop()
        #Mettre la page dans la liste
        listes_pages.append(echantillon)

    for key in corpus.keys():
        pageranks[key] = listes_pages.count(key)/n
    return pageranks

"""
#Vérifier que ma fonction sample_pagerank fonctionne coorectement
corpus = {"1.html" : {"2.html", "3.html"}, "2.html" : {"3.html"},"3.html" : {"2.html"}}
print(sample_pagerank(corpus, 0.85, 1000))
"""

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #raise NotImplementedError
    import numpy as np
    
    #Mettre les clés de corpus dans une lsite
    corpus_keys = list(corpus.keys())
    #Nombre de pages
    N_pages = len(corpus_keys)
    
    #Initialisation: Attribuer chaque page un rang 1/N
    dist_proba = {key:1/N_pages for key in corpus_keys}

    
    #Calculer de nouvelles valeurs de classement et répeter
    while True:
        
        #Mettre la valeur actuelle dans precedent pour ensuite mettre à jour
        precedent = dict(dist_proba)
        #Initialiser notre condition d'arrêt
        condition = np.array([])
        
        #On calcul de nouvelles valeurs pour chaque page
        for page in corpus_keys:
            #On initialise la somme qui nous permettra par la suite de calculer
            #la formule du PR donné dans le contexte du projet
            somme = 0
            #on parcourt à nouveau le corpus pour calculer la somme
            for parent in corpus_keys:           
                if page in corpus[parent]:
                    somme += precedent[parent]/len(corpus[parent])
                elif len(corpus[parent]) == 0:
                    somme += 1/N_pages
                #On calcule le pagerank de chaque page selon la formule
                #PR(p) = (1-d)/N + d*somme(PR(i)/Num_links(i))
                dist_proba[page] = (1-damping_factor)/N_pages + (damping_factor * somme)
                #On récupère la différence entre la page précédente et actuelle
                condition = np.append(condition, [abs(precedent[page] - dist_proba[page])])
        #Condition d'arrêt        
        if (condition < 0.001).any():
            return dist_proba

    return dist_proba

"""
#Vérifier que ma fonction transition_model fonctionne coorectlement
corpus = {"1.html" : {"2.html", "3.html"}, "2.html" : {"3.html"},"3.html" : {"2.html"}}
iterate_pagerank(corpus, 0.85)
"""

if __name__ == "__main__":
    main()
