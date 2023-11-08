# Exercise 1.
# Using Entrez API to Fetch PubMed Data

The Entrez API provides programmatic access to various biomedical databases hosted by the National Center for Biotechnology Information (NCBI). In this task, we aim to retrieve the metadata of 1000 Alzheimer's papers and 1000 cancer papers from 2023 available in the PubMed database.

## Steps to Achieve the Task:

1. **Search for Papers**: Use the Entrez API to search for papers based on specific queries and retrieve their PubMed IDs.
2. **Fetch Paper Metadata**: For each retrieved PubMed ID, fetch the paper's metadata.
3. **Parse and Save Metadata**: Parse the fetched metadata to extract the required information and save it in a JSON format.


```python
import requests
import time
import xml.dom.minidom as m
import json
```


```python
def get_id(disease):
    r = requests.get(
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    f"db=pubmed&term={disease}+AND+2023[pdat]&retmode=xml&retmax=1000"
  )
    time.sleep(1)
    if r.status_code == 200:
        doc = m.parseString(r.text)
        IDs = doc.getElementsByTagName("Id")
        pubmed_id = [ID.childNodes[0].data for ID in IDs]
        return pubmed_id
    else:
        print("Failed to fetch IDs:", r.status_code)
        return []
```


```python
def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
        else:
            rc.append(getText(node.childNodes))
    return ''.join(rc)
```


```python
def get_metadata(id_list, query_term):
    metadata_dict = {}
    if id_list:
        r = requests.post(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            data={
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(id_list)
            }
        )
        time.sleep(1)
        if r.status_code == 200:
            doc = m.parseString(r.text)
            articles = doc.getElementsByTagName("PubmedArticle")
            for article, pubmed_id in zip(articles, id_list):
                title_node = article.getElementsByTagName("ArticleTitle")
                abstract_nodes = article.getElementsByTagName("AbstractText")
                title = getText(title_node[0].childNodes) if title_node else "N/A"
                abstract = " ".join(getText(abstract_node.childNodes) for abstract_node in abstract_nodes) if abstract_nodes else "N/A"
                metadata_dict[pubmed_id] = {
                    "ArticleTitle": title,
                    "AbstractText": abstract,
                    "query": query_term
                }
        else:
            print("Failed to fetch metadata:", r.status_code)
    return metadata_dict
```


```python
if __name__ == "__main__":
    Alzheimers_id = get_id("Alzheimers")
    Cancer_id = get_id("cancer")

    common_ids = list(set(Alzheimers_id) & set(Cancer_id))
    print("Common IDs:", common_ids)

    Alzheimers_metadata = get_metadata(Alzheimers_id, "Alzheimers")
    Cancer_metadata = get_metadata(Cancer_id, "cancer")

    all_metadata = {**Alzheimers_metadata, **Cancer_metadata}
    with open('metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=4)
    print("Metadata saved to 'metadata.json'")
```

    Common IDs: ['37895928', '37897137', '37895969', '37901920', '37902389', '37899058']
    Metadata saved to 'metadata.json'


I identifies that there is some overlap between the two sets of papers (Alzheimer's and cancer papers) as `common_ids`

### Regarding the handling of multiple AbstractText fields:
**Concatenating with a Space**: This is a straightforward approach and makes it easy to read and process the abstract as a single string later. However, it might not preserve the structure of the original abstract, which could be important for understanding the flow or sections of the abstract.

# Exercise 2.
# Computing SPECTER Embeddings for Papers

Machine learning and data visualization strategies generally work best on data that is numeric. However, text data is also quite common in various domains. Modern Natural Language Processing (NLP) algorithms powered by machine learning, and trained on massive datasets, can convert text data into numeric vectors. Such algorithms ensure that similar items are represented by similar vectors.

For this task, we will use the SPECTER model to compute embeddings for paper titles and abstracts identified in a previous exercise. The SPECTER model returns a 768-dimensional vector for each text input.


```python
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')
```

    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    pytorch_model.bin: 100%|██████████| 440M/440M [01:46<00:00, 4.15MB/s] 



```python
import json
with open('metadata.json') as f:
    papers = json.load(f)
```


```python
import tqdm

# we can use a persistent dictionary (via shelve) so we can stop and restart if needed
# alternatively, do the same but with embeddings starting as an empty dictionary
embeddings = {}
for pmid, paper in tqdm.tqdm(papers.items()):
    data = [paper["ArticleTitle"] + tokenizer.sep_token + "".join(paper["AbstractText"])]
    inputs = tokenizer(
        data, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

# turn our dictionary into a list
embeddings = [embeddings[pmid] for pmid in papers.keys()]
```

    100%|██████████| 1994/1994 [32:06<00:00,  1.03it/s]



```python
from sklearn import decomposition
import pandas as pd

pca = decomposition.PCA(n_components=3)
embeddings_pca = pd.DataFrame(
    pca.fit_transform(embeddings),
    columns=['PC0', 'PC1', 'PC2']
)
embeddings_pca["query"] = [paper["query"] for paper in papers.values()]
```


```python
embeddings_pca
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC0</th>
      <th>PC1</th>
      <th>PC2</th>
      <th>query</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.499764</td>
      <td>3.079033</td>
      <td>-2.893260</td>
      <td>Alzheimers</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-5.236744</td>
      <td>4.390724</td>
      <td>1.171794</td>
      <td>Alzheimers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-5.105960</td>
      <td>-2.530446</td>
      <td>-3.978613</td>
      <td>Alzheimers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-6.976255</td>
      <td>-2.731421</td>
      <td>-1.645280</td>
      <td>Alzheimers</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-3.770983</td>
      <td>-1.909597</td>
      <td>1.119820</td>
      <td>Alzheimers</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>3.519124</td>
      <td>-4.601136</td>
      <td>1.202660</td>
      <td>cancer</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>2.226133</td>
      <td>-2.481808</td>
      <td>3.815585</td>
      <td>cancer</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>3.235941</td>
      <td>0.327157</td>
      <td>-0.078199</td>
      <td>cancer</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>4.228046</td>
      <td>-1.301272</td>
      <td>0.139682</td>
      <td>cancer</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>4.158810</td>
      <td>-2.885532</td>
      <td>2.754484</td>
      <td>cancer</td>
    </tr>
  </tbody>
</table>
<p>1994 rows × 4 columns</p>
</div>




```python
import matplotlib.pyplot as plt
```


```python
# Plot 2D scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PC0 vs PC1
axes[0].scatter(embeddings_pca["PC0"], embeddings_pca["PC1"], c=embeddings_pca["query"].map({"Alzheimers": "red", "cancer": "blue"}))
axes[0].set_title('PC0 vs PC1')
axes[0].set_xlabel('PC0')
axes[0].set_ylabel('PC1')

# PC0 vs PC2
axes[1].scatter(embeddings_pca["PC0"], embeddings_pca["PC2"], c=embeddings_pca["query"].map({"Alzheimers": "red", "cancer": "blue"}))
axes[1].set_title('PC0 vs PC2')
axes[1].set_xlabel('PC0')
axes[1].set_ylabel('PC2')

# PC1 vs PC2
axes[2].scatter(embeddings_pca["PC1"], embeddings_pca["PC2"], c=embeddings_pca["query"].map({"Alzheimers": "red", "cancer": "blue"}))
axes[2].set_title('PC1 vs PC2')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')

plt.show()
```


    
![png](output_14_0.png)
    


Observations from the scatter plots:

- **PC0 vs PC1**:
  1. Two distinct clusters are evident, with some overlap.
  2. The red cluster is more elongated along the PC0 axis, while the blue cluster has a broader spread along the PC1 axis.

- **PC0 vs PC2**:
  1. The separation between clusters is less pronounced compared to the PC0 vs PC1 plot.
  2. Both clusters exhibit a vertical spread along the PC2 axis, with the blue cluster being more densely packed.

- **PC1 vs PC2**:
  1. There is significant overlap, making differentiation challenging.
  2. Both red and blue points are interspersed without clear separation.

Overall, PC0 vs PC1 provides the most distinct separation between the two groups, while PC1 vs PC2 offers the least.


# Exercise 3
## Gradient Descent for Optimizing Parameters `a` and `b`
To implement the gradient descent algorithm for this problem, follow these steps:
1. Query the given API to get the error value for a set of parameters `(a, b)`.
2. Compute the gradient of the error with respect to both parameters.
3. Update the parameters `a` and `b` in the direction that reduces the error.
4. Repeat the above steps until the error converges to a minimum value or after a certain number of iterations.


```python
import requests

def get_error(a, b):
    """Query the API to get error for the given a and b values."""
    url = f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}"
    return float(requests.get(url, headers={"User-Agent": "MyScript"}).text)
```


```python
def gradient_descent(a_start=0.5, b_start=0.5, learning_rate=0.1, iterations=100, delta=0.01):
    """Perform 2D gradient descent."""
    a = a_start
    b = b_start
    
    for i in range(iterations):
        # Calculate gradient
        error_current = get_error(a, b)
        
        a_gradient = (get_error(a + delta, b) - error_current) / delta
        b_gradient = (get_error(a, b + delta) - error_current) / delta
        
        # Update a and b
        a = a - learning_rate * a_gradient
        b = b - learning_rate * b_gradient
        
        # Print the error for current iteration
        print(f"Iteration {i+1}: Error = {error_current}, a = {a}, b = {b}")
        
    return a, b

```


```python
# Run gradient descent
optimal_a, optimal_b = gradient_descent()
print(f"Optimal values: a = {optimal_a}, b = {optimal_b}")
```

    Iteration 1: Error = 1.216377, a = 0.44220000000000104, b = 0.5368000000000013
    Iteration 2: Error = 1.17433128, a = 0.3959600000000014, b = 0.5662400000000019
    Iteration 3: Error = 1.1474556192, a = 0.3589680000000013, b = 0.5897920000000019
    Iteration 4: Error = 1.13028207629, a = 0.3293744000000016, b = 0.6086336000000032
    Iteration 5: Error = 1.11931251282, a = 0.3056995200000019, b = 0.6237068800000052
    Iteration 6: Error = 1.11230919541, a = 0.28675961600000033, b = 0.635765504000005
    Iteration 7: Error = 1.10784083482, a = 0.2716076927999995, b = 0.6454124032000048
    Iteration 8: Error = 1.10499209409, a = 0.2594861541999989, b = 0.6531299225000038
    Iteration 9: Error = 1.10317770807, a = 0.24978892340000014, b = 0.6593039380000039
    Iteration 10: Error = 1.10202354744, a = 0.24203113870000026, b = 0.664243150400003
    Iteration 11: Error = 1.10129052178, a = 0.23582491089999946, b = 0.6681945203000024
    Iteration 12: Error = 1.10082589508, a = 0.23085992869999972, b = 0.6713556163000041
    Iteration 13: Error = 1.10053214176, a = 0.22688794299999993, b = 0.673884493100005
    Iteration 14: Error = 1.10034702585, a = 0.22371035440000142, b = 0.6759075945000066
    Iteration 15: Error = 1.10023086065, a = 0.2211682836000013, b = 0.6775260756000074
    Iteration 16: Error = 1.1001583621, a = 0.21913462690000118, b = 0.6788208605000063
    Iteration 17: Error = 1.10011344077, a = 0.2175077016000011, b = 0.6798566884000063
    Iteration 18: Error = 1.10008587331, a = 0.2162061613000028, b = 0.6806853507000072
    Iteration 19: Error = 1.1000691759, a = 0.21516492910000418, b = 0.6813482806000088
    Iteration 20: Error = 1.10005924615, a = 0.21433194320000526, b = 0.6818786244000092
    Iteration 21: Error = 1.1000534964, a = 0.2136655545000039, b = 0.6823028995000087
    Iteration 22: Error = 1.10005030079, a = 0.21313244360000327, b = 0.6826423196000082
    Iteration 23: Error = 1.10004864298, a = 0.21270595490000277, b = 0.6829138557000087
    Iteration 24: Error = 1.10004789189, a = 0.21236476400000326, b = 0.68313108460001
    Iteration 25: Error = 1.10004765911, a = 0.2120918112000032, b = 0.6833048677000093
    Iteration 26: Error = 1.10004770847, a = 0.2118734489000036, b = 0.6834438941000083
    Iteration 27: Error = 1.10004789874, a = 0.21169875920000258, b = 0.683555115300007
    Iteration 28: Error = 1.10004814744, a = 0.21155900730000132, b = 0.683644092200006
    Iteration 29: Error = 1.10004840816, a = 0.2114472058000012, b = 0.6837152737000047
    Iteration 30: Error = 1.10004865627, a = 0.21135776470000156, b = 0.6837722190000046
    Iteration 31: Error = 1.10004888004, a = 0.21128621170000095, b = 0.6838177752000036
    Iteration 32: Error = 1.10004907525, a = 0.2112289693000009, b = 0.6838542201000037
    Iteration 33: Error = 1.10004924178, a = 0.2111831754000022, b = 0.6838833760000047
    Iteration 34: Error = 1.10004938164, a = 0.21114654030000413, b = 0.683906700800005
    Iteration 35: Error = 1.10004949777, a = 0.2111172323000039, b = 0.6839253607000035
    Iteration 36: Error = 1.10004959338, a = 0.2110937858000046, b = 0.6839402885000041
    Iteration 37: Error = 1.10004967162, a = 0.2110750287000025, b = 0.6839522308000041
    Iteration 38: Error = 1.10004973532, a = 0.21106002300000215, b = 0.6839617847000041
    Iteration 39: Error = 1.10004978699, a = 0.21104801840000142, b = 0.6839694278000037
    Iteration 40: Error = 1.10004982878, a = 0.21103841470000173, b = 0.6839755423000033
    Iteration 41: Error = 1.1000498625, a = 0.21103073170000108, b = 0.6839804338000017
    Iteration 42: Error = 1.10004988967, a = 0.211024585300001, b = 0.6839843470000013
    Iteration 43: Error = 1.10004991153, a = 0.21101966830000052, b = 0.6839874776000006
    Iteration 44: Error = 1.10004992908, a = 0.21101573460000145, b = 0.6839899820000013
    Iteration 45: Error = 1.10004994318, a = 0.21101258770000086, b = 0.6839919856000005
    Iteration 46: Error = 1.10004995449, a = 0.2110100702000013, b = 0.6839935884999999
    Iteration 47: Error = 1.10004996356, a = 0.21100805620000118, b = 0.6839948708000017
    Iteration 48: Error = 1.10004997082, a = 0.2110064449000011, b = 0.6839958966000022
    Iteration 49: Error = 1.10004997664, a = 0.21100515590000102, b = 0.6839967172000034
    Iteration 50: Error = 1.10004998131, a = 0.21100412480000008, b = 0.6839973738000023
    Iteration 51: Error = 1.10004998504, a = 0.21100329990000066, b = 0.6839978991000035
    Iteration 52: Error = 1.10004998803, a = 0.21100264000000157, b = 0.683998319300005
    Iteration 53: Error = 1.10004999042, a = 0.2110021120000023, b = 0.6839986555000062
    Iteration 54: Error = 1.10004999233, a = 0.21100168960000287, b = 0.6839989244000062
    Iteration 55: Error = 1.10004999386, a = 0.21100135160000155, b = 0.6839991395000062
    Iteration 56: Error = 1.10004999509, a = 0.2110010813000014, b = 0.6839993116000072
    Iteration 57: Error = 1.10004999607, a = 0.21100086500000126, b = 0.6839994493000074
    Iteration 58: Error = 1.10004999686, a = 0.21100069200000027, b = 0.6839995595000077
    Iteration 59: Error = 1.10004999749, a = 0.21100055359999992, b = 0.6839996476000083
    Iteration 60: Error = 1.10004999799, a = 0.21100044289999964, b = 0.6839997181000075
    Iteration 61: Error = 1.10004999839, a = 0.2110003543000012, b = 0.6839997745000077
    Iteration 62: Error = 1.10004999871, a = 0.211000283400002, b = 0.6839998196000092
    Iteration 63: Error = 1.10004999897, a = 0.21100022670000396, b = 0.68399985570001
    Iteration 64: Error = 1.10004999918, a = 0.21100018140000243, b = 0.6839998846000079
    Iteration 65: Error = 1.10004999934, a = 0.21100014510000165, b = 0.6839999077000076
    Iteration 66: Error = 1.10004999947, a = 0.21100011610000147, b = 0.6839999261000069
    Iteration 67: Error = 1.10004999958, a = 0.21100009290000177, b = 0.6839999409000059
    Iteration 68: Error = 1.10004999966, a = 0.21100007430000023, b = 0.6839999527000047
    Iteration 69: Error = 1.10004999973, a = 0.211000059399999, b = 0.6839999622000033
    Iteration 70: Error = 1.10004999978, a = 0.211000047499998, b = 0.6839999697000039
    Iteration 71: Error = 1.10004999983, a = 0.21100003799999723, b = 0.6839999758000044
    Iteration 72: Error = 1.10004999986, a = 0.2110000303999966, b = 0.6839999806000048
    Iteration 73: Error = 1.10004999989, a = 0.2110000242999961, b = 0.6839999845000051
    Iteration 74: Error = 1.10004999991, a = 0.2110000193999957, b = 0.6839999876000054
    Iteration 75: Error = 1.10004999993, a = 0.21100001549999536, b = 0.6839999901000056
    Iteration 76: Error = 1.10004999994, a = 0.2110000123999951, b = 0.6839999920000057
    Iteration 77: Error = 1.10004999996, a = 0.2110000099999949, b = 0.6839999936000059
    Iteration 78: Error = 1.10004999996, a = 0.21100000799999474, b = 0.683999994800006
    Iteration 79: Error = 1.10004999997, a = 0.2110000063999946, b = 0.683999995800006
    Iteration 80: Error = 1.10004999998, a = 0.2110000050999945, b = 0.6839999967000061
    Iteration 81: Error = 1.10004999998, a = 0.21100000409999442, b = 0.6839999973000062
    Iteration 82: Error = 1.10004999999, a = 0.21100000329999435, b = 0.6839999979000062
    Iteration 83: Error = 1.10004999999, a = 0.2110000026999943, b = 0.6839999983000062
    Iteration 84: Error = 1.10004999999, a = 0.21100000219999426, b = 0.6839999986000063
    Iteration 85: Error = 1.10004999999, a = 0.21100000169999422, b = 0.6839999989000063
    Iteration 86: Error = 1.10004999999, a = 0.2110000012999942, b = 0.6839999991000063
    Iteration 87: Error = 1.10005, a = 0.21100000109999417, b = 0.6839999993000063
    Iteration 88: Error = 1.10005, a = 0.21100000089999416, b = 0.6839999995000063
    Iteration 89: Error = 1.10005, a = 0.21100000079999415, b = 0.6839999996000063
    Iteration 90: Error = 1.10005, a = 0.21100000069999414, b = 0.6839999997000064
    Iteration 91: Error = 1.10005, a = 0.21100000059999413, b = 0.6839999998000064
    Iteration 92: Error = 1.10005, a = 0.21100000049999412, b = 0.6839999999000064
    Iteration 93: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 94: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 95: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 96: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 97: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 98: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 99: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Iteration 100: Error = 1.10005, a = 0.21100000039999411, b = 0.6840000000000064
    Optimal values: a = 0.21100000039999411, b = 0.6840000000000064


### Estimation of the Gradient
In the absence of an explicit formula to compute the derivative, we estimate the gradient using a method called the finite difference method. Specifically, we use the `forward difference` approximation for the partial derivatives:
1. **For `a`:**
$$ \frac{\partial \text{Error}}{\partial a} = \frac{\text{Error}(a+\delta, b) - \text{Error}(a,b)}{\delta} $$

2. **For `b`:**
$$ \frac{\partial \text{Error}}{\partial b} = \frac{\text{Error}(a, b+\delta) - \text{Error}(a,b)}{\delta} $$

Here, $ \delta $ is a small positive number that helps approximate the slope of the error function at a given point `(a, b)`.

### Numerical Choices:

1. **Initial Values `a_start=0.5` and `b_start=0.5`:** These are the starting values for \(a\) and \(b\). Starting at the midpoint of the allowed parameter range seemed like a neutral choice, but depending on prior knowledge or other considerations, different starting points could be chosen.

2. **Learning Rate `learning_rate=0.1`:** This determines the step size in the direction of the negative gradient. A smaller learning rate might converge more reliably but slower, whereas a larger learning rate might converge faster but risks overshooting the minimum.

3. **Iterations `iterations=100`:** This is the number of times the algorithm will update the parameters. This choice means we are allowing the algorithm up to 100 updates to find the optimal parameters. This is an arbitrary choice and in practice, might be set based on when the changes in error or parameters become negligibly small.

4. **Delta `delta=0.01`:** This small value is used to approximate the gradient. The choice of $ \delta $ represents a trade-off: a smaller $ \delta $ might give a more accurate approximation of the gradient but could be more susceptible to numerical errors, while a larger $ \delta $ might be less accurate but more stable.

### Justifications:

1. **Gradient Estimation:** The forward difference method provides a simple and intuitive way to estimate the gradient. It's essentially measuring the "rise over run" over a very short distance, which approximates the instantaneous rate of change.

2. **Learning Rate:** The chosen value is a commonly used starting point in gradient descent. It's a middle-ground choice that's neither too small nor too large. However, in practice, this might be tuned based on the problem.

3. **Iterations:** While 100 iterations is an arbitrary choice, it often suffices for many problems. In a more refined version, one might implement a convergence criterion, like if the difference in error between successive iterations is below a certain threshold.

4. **Delta:** The value of 0.01 for $ \delta $ is a typical choice for numerical differentiation in the unit interval [0, 1]. It's a balance between accuracy and stability. Too small a $ \delta $ could lead to numerical instability, while too large a $ \delta $ could lead to inaccurate gradient estimates.

To find both the local and global minima, we can use the gradient descent method, as previously discussed. However, we need to address the challenge of the presence of multiple minima in our error surface. 

1. **Multiple Starting Points**: To find both local and global minima, we will run the gradient descent algorithm from different initial values of `(a, b)`. The idea is that, depending on our starting point, we may converge to different minima.

2. **Determining Local vs. Global Minima**: Once we have the minima locations, we can compare the error values at these points. The one with the lowest error is the global minimum, and the other is the local minimum.

3. **Validation for Local vs. Global Minima**: If we did not know how many minima were present, we would have used techniques like:
    - **Grid Search**: A systematic search through a subset of the parameter space, while not exhaustive, can give an idea of regions of interest that may contain minima.
    - **Random Restart**: We would run gradient descent multiple times with random initial values for `(a, b)`, and note the different minima we arrive at. If we keep arriving at the same minimum, it’s likely global, but if we find multiple, it indicates the presence of multiple minima.
    - **Higher Order Derivatives**: A second-order derivative or the Hessian can be useful. A positive value indicates a local minimum, while a negative value indicates a local maximum. However, computing this for complex functions can be challenging.

I'll now query the API from multiple starting points to find both the local and global minima. Let's implement this. 


```python
import requests
import numpy as np

# Define the error function based on API query
def get_error(a, b):
    return float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}", 
                              headers={"User-Agent": "GradientDescentMinFinder"}).text)

# Define the gradient estimation
def gradient(a, b, delta=0.01):
    dE_da = (get_error(a + delta, b) - get_error(a, b)) / delta
    dE_db = (get_error(a, b + delta) - get_error(a, b)) / delta
    return dE_da, dE_db

# Gradient Descent
def gradient_descent(a_start, b_start, learning_rate=0.1, max_iters=100, tolerance=1e-6):
    a, b = a_start, b_start
    for i in range(max_iters):
        dE_da, dE_db = gradient(a, b)
        a -= learning_rate * dE_da
        b -= learning_rate * dE_db
        
        # Stopping criteria
        if np.sqrt(dE_da**2 + dE_db**2) < tolerance:
            break
    return a, b

# Using multiple starting points to find minima
starting_points = [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1)]
minima = [gradient_descent(a, b) for a, b in starting_points]

# Checking error at found minima to determine global vs. local
errors = [get_error(a, b) for a, b in minima]
global_minimum = minima[np.argmin(errors)]
local_minimum = minima[np.argmax(errors)]

print(f"Global Minimum: {global_minimum}")
print(f"Local Minimum: {local_minimum}")
```

    Global Minimum: (0.7070000332000018, 0.16399998909999428)
    Local Minimum: (0.21099993040000423, 0.683999633400003)


# Exercise 4
To analyze the time complexity of the `merge_sort` function you provided, we will do the following:

1. Generate a sequence of array sizes `n` using `numpy.logspace`. This will allow us to have sizes that are evenly spaced on a log scale.
2. For each size `n`, generate random data of that size.
3. Time the `merge_sort` function using `time.perf_counter()` for each array size.
4. Plot the results on a log-log graph to examine the apparent big-O scaling.


```python
import numpy as np
import matplotlib.pyplot as plt
import time
```


```python
def merge_sort(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(merge_sort(data[:split]))
        right = iter(merge_sort(data[split:]))
        result = []
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top:
                result.append(left_top)
                try:
                    left_top = next(left)
                except StopIteration:
                    return result + [right_top] + list(right)
            else:
                result.append(right_top)
                try:
                    right_top = next(right)
                except StopIteration:
                    return result + [left_top] + list(left)
```


```python
# Define sizes that are evenly spaced on a log scale
sizes = np.logspace(1, 10, num=15, base=2, dtype=int)
```


```python
# Store timings
times = []

for n in sizes:
    data = np.random.rand(n)  # generate random data
    start_time = time.perf_counter()
    merge_sort(data)
    end_time = time.perf_counter()
    times.append(end_time - start_time)
```


```python
# Plot results on a log-log graph
plt.loglog(sizes, times, 'o-', basex=2, basey=2)
plt.xlabel('Size n')
plt.ylabel('Time (s)')
plt.title('Performance of merge_sort')
plt.grid(True, which="both", ls="--", c='0.65')
plt.show()
```

    /var/folders/94/cs6rph4n3495kjmx9rh4kv6h0000gn/T/ipykernel_59466/1695494882.py:2: MatplotlibDeprecationWarning: The 'basex' parameter of __init__() has been renamed 'base' since Matplotlib 3.3; support for the old name will be dropped two minor releases later.
      plt.loglog(sizes, times, 'o-', basex=2, basey=2)
    /var/folders/94/cs6rph4n3495kjmx9rh4kv6h0000gn/T/ipykernel_59466/1695494882.py:2: MatplotlibDeprecationWarning: The 'basey' parameter of __init__() has been renamed 'base' since Matplotlib 3.3; support for the old name will be dropped two minor releases later.
      plt.loglog(sizes, times, 'o-', basex=2, basey=2)



    
![png](output_30_1.png)
    


The graph displays a behavior consistent with $ O(n \log n) $ complexity, typical for merge sort. In a log-log scale, this complexity presents as a linear relationship, which we observe in the provided plot.

To implement a parallel version of `merge_sort` using the `multiprocessing` module, we'll split the data in half, then use two separate processes to sort each half. Finally, we'll merge the two sorted halves together.

Here's a parallel version of the `merge_sort` function using multiprocessing:


```python
import multiprocess

def parallel_merge_sort(data):
    if len(data) <= 1:
        return data

    split = len(data) // 2
    left = data[:split]
    right = data[split:]

    with multiprocess.Pool(2) as pool:
        left, right = pool.map(merge_sort, [left, right])

    return merge(left, right)

def merge(left, right):
    result = []
    left_index, right_index = 0, 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1


    result.extend(left[left_index:])
    result.extend(right[right_index:])
    return result

# Traditional merge_sort for comparison
def merge_sort(data):
    if len(data) <= 1:
        return data

    split = len(data) // 2
    left = merge_sort(data[:split])
    right = merge_sort(data[split:])
    return merge(left, right)
```

To compare the performance:


```python
# Loop to run merge_sort and parallel_merge_sort for different sizes of n
for exponent in range(1, 8):
    n = 10 ** exponent
    data = np.random.randint(0, n, n).tolist()

    print(f"\nSorting {n} elements:")

    # Traditional merge_sort
    start_time = time.perf_counter()
    sorted_data = merge_sort(data.copy())
    end_time = time.perf_counter()
    print(f"Traditional merge_sort took {end_time - start_time:.4f} seconds.")

    # Parallel merge_sort
    start_time = time.perf_counter()
    sorted_data_parallel = parallel_merge_sort(data.copy())
    end_time = time.perf_counter()
    print(f"Parallel merge_sort took {end_time - start_time:.4f} seconds.")
 
```

    
    Sorting 10 elements:
    Traditional merge_sort took 0.9755 seconds.
    Parallel merge_sort took 0.8761 seconds.
    
    Sorting 100 elements:
    Traditional merge_sort took 0.0002 seconds.
    Parallel merge_sort took 0.0292 seconds.
    
    Sorting 1000 elements:
    Traditional merge_sort took 0.0025 seconds.
    Parallel merge_sort took 0.0253 seconds.
    
    Sorting 10000 elements:
    Traditional merge_sort took 0.0323 seconds.
    Parallel merge_sort took 0.0595 seconds.
    
    Sorting 100000 elements:
    Traditional merge_sort took 0.3949 seconds.
    Parallel merge_sort took 0.4486 seconds.
    
    Sorting 1000000 elements:
    Traditional merge_sort took 4.8799 seconds.
    Parallel merge_sort took 4.8204 seconds.
    
    Sorting 10000000 elements:
    Traditional merge_sort took 60.9296 seconds.
    Parallel merge_sort took 54.5377 seconds.


### Discussion:
1. **Performance**: The parallel version should generally be faster than the traditional version for sufficiently large `n`, especially if there are multiple CPU cores available.
2. **Overhead**: There's some overhead associated with creating processes and communicating between them. If `n` is small, this overhead might make the parallel version slower than the traditional one.
3. **Limitations**: The implementation uses a fixed number of processes (2). For even better performance, a dynamic number of processes based on the size of `n` or the number of available CPU cores could be used.
4. **Memory**: The parallel version might use more memory, as it needs to maintain multiple processes.

# Exercise 5
## Dataset: **MIMIC-III** (Medical Information Mart for Intensive Care III)
- The dataset is publicly available but requires researchers to sign a data use agreement due to the sensitive nature of medical data. It is released under the [Health Insurance Portability and Accountability Act (HIPAA)](https://www.hhs.gov/hipaa/index.html).


```python
import pandas as pd

# Load the dataset into a pandas DataFrame
admissions_df = pd.read_csv('ADMISSIONS.csv')

# Display the first few rows of the dataset to verify it loaded correctly
print(admissions_df.head())
```

       row_id  subject_id  hadm_id            admittime            dischtime  \
    0   12258       10006   142345  2164-10-23 21:09:00  2164-11-01 17:15:00   
    1   12263       10011   105331  2126-08-14 22:32:00  2126-08-28 18:59:00   
    2   12265       10013   165520  2125-10-04 23:36:00  2125-10-07 15:13:00   
    3   12269       10017   199207  2149-05-26 17:19:00  2149-06-03 18:42:00   
    4   12270       10019   177759  2163-05-14 20:43:00  2163-05-15 12:00:00   
    
                 deathtime admission_type         admission_location  \
    0                  NaN      EMERGENCY       EMERGENCY ROOM ADMIT   
    1  2126-08-28 18:59:00      EMERGENCY  TRANSFER FROM HOSP/EXTRAM   
    2  2125-10-07 15:13:00      EMERGENCY  TRANSFER FROM HOSP/EXTRAM   
    3                  NaN      EMERGENCY       EMERGENCY ROOM ADMIT   
    4  2163-05-15 12:00:00      EMERGENCY  TRANSFER FROM HOSP/EXTRAM   
    
      discharge_location insurance language  religion marital_status  \
    0   HOME HEALTH CARE  Medicare      NaN  CATHOLIC      SEPARATED   
    1       DEAD/EXPIRED   Private      NaN  CATHOLIC         SINGLE   
    2       DEAD/EXPIRED  Medicare      NaN  CATHOLIC            NaN   
    3                SNF  Medicare      NaN  CATHOLIC       DIVORCED   
    4       DEAD/EXPIRED  Medicare      NaN  CATHOLIC       DIVORCED   
    
                    ethnicity            edregtime            edouttime  \
    0  BLACK/AFRICAN AMERICAN  2164-10-23 16:43:00  2164-10-23 23:00:00   
    1   UNKNOWN/NOT SPECIFIED                  NaN                  NaN   
    2   UNKNOWN/NOT SPECIFIED                  NaN                  NaN   
    3                   WHITE  2149-05-26 12:08:00  2149-05-26 19:45:00   
    4                   WHITE                  NaN                  NaN   
    
                 diagnosis  hospital_expire_flag  has_chartevents_data  
    0               SEPSIS                     0                     1  
    1          HEPATITIS B                     1                     1  
    2               SEPSIS                     1                     1  
    3     HUMERAL FRACTURE                     0                     1  
    4  ALCOHOLIC HEPATITIS                     1                     1  



```python
# Generate descriptive statistics for numerical columns
numerical_stats = admissions_df.describe()

# Check for missing values
missing_values = admissions_df.isnull().sum()

numerical_stats, missing_values
```




    (             row_id    subject_id        hadm_id  hospital_expire_flag  \
     count    129.000000    129.000000     129.000000            129.000000   
     mean   28036.441860  28010.410853  152343.441860              0.310078   
     std    14036.548988  16048.502883   27858.788248              0.464328   
     min    12258.000000  10006.000000  100375.000000              0.000000   
     25%    12339.000000  10088.000000  128293.000000              0.000000   
     50%    39869.000000  40310.000000  157235.000000              0.000000   
     75%    40463.000000  42135.000000  174739.000000              1.000000   
     max    41092.000000  44228.000000  199395.000000              1.000000   
     
            has_chartevents_data  
     count            129.000000  
     mean               0.992248  
     std                0.088045  
     min                0.000000  
     25%                1.000000  
     50%                1.000000  
     75%                1.000000  
     max                1.000000  ,
     row_id                   0
     subject_id               0
     hadm_id                  0
     admittime                0
     dischtime                0
     deathtime               89
     admission_type           0
     admission_location       0
     discharge_location       0
     insurance                0
     language                48
     religion                 1
     marital_status          16
     ethnicity                0
     edregtime               37
     edouttime               37
     diagnosis                0
     hospital_expire_flag     0
     has_chartevents_data     0
     dtype: int64)



From the descriptive statistics, we can see that the `hospital_expire_flag` column, which indicates whether the patient expired in the hospital, has a mean of approximately 0.31. This suggests that about 31% of the admissions in this dataset resulted in the patient's death in the hospital. The `has_chartevents_data` column has a mean of approximately 0.99, indicating that chart events data is available for about 99% of the admissions.

The missing values count shows that there are missing values in the `deathtime`, `language`, and `religion` columns. The `deathtime` column has 89 missing values, which is expected as not all admissions result in the patient's death. The `language` column has 48 missing values, and the `religion` column has 1 missing value. These missing values may need to be addressed depending on the specific analysis to be performed on the data.

Next, let's visualize some of the data to gain further insights. We'll create a bar plot of the `admission_type` column to see the distribution of admission types


```python
import matplotlib.pyplot as plt

# Create a bar plot of the 'admission_type' column
counts = admissions_df['admission_type'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(counts.index, counts.values, color='skyblue')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.title('Distribution of Admission Types')
plt.show()
```


    
![png](output_42_0.png)
    


From this plot, we can see that **EMERGENCY** is the most common admission type, followed by **ELECTIVE** and **URGENT**.

Here is the average duration of the hospital stay for each admission type:


```python

# Convert 'admittime' and 'dischtime' to datetime
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

# Calculate the duration of the hospital stay
admissions_df['stay_duration'] = (admissions_df['dischtime'] - admissions_df['admittime']).dt.total_seconds() / 86400  # convert to days

# Calculate the average stay duration for each admission type
average_stay_duration = admissions_df.groupby('admission_type')['stay_duration'].mean()

average_stay_duration
```




    admission_type
    ELECTIVE     11.668403
    EMERGENCY     9.226932
    URGENT        6.259375
    Name: stay_duration, dtype: float64



As you can see, the average stay duration is longest for elective admissions, followed by emergency admissions, and then urgent admissions.

Here are the histograms showing the distribution of the duration of the hospital stay for each type of admission:


```python
# Create histograms of the stay duration for each admission type
plt.figure(figsize=(15, 10))

# Elective admissions
plt.subplot(3, 1, 1)
plt.hist(admissions_df[admissions_df['admission_type'] == 'ELECTIVE']['stay_duration'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Stay Duration (days)')
plt.ylabel('Count')
plt.title('Stay Duration for Elective Admissions')

# Emergency admissions
plt.subplot(3, 1, 2)
plt.hist(admissions_df[admissions_df['admission_type'] == 'EMERGENCY']['stay_duration'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Stay Duration (days)')
plt.ylabel('Count')
plt.title('Stay Duration for Emergency Admissions')

# Urgent admissions
plt.subplot(3, 1, 3)
plt.hist(admissions_df[admissions_df['admission_type'] == 'URGENT']['stay_duration'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Stay Duration (days)')
plt.ylabel('Count')
plt.title('Stay Duration for Urgent Admissions')

plt.tight_layout()
plt.show()
```


    
![png](output_48_0.png)
    


From these histograms, we can see the distribution of stay durations for elective, emergency, and urgent admissions. The distributions are skewed to the right, indicating that most stays are relatively short, but there are a few longer stays that pull the mean to the right.

In terms of data cleaning, we identified earlier that there are missing values in the `deathtime`, `language`, and `religion` columns. Depending on the specific analysis to be performed on the data, these missing values may need to be addressed. For instance, if we are interested in analyzing the data by language or religion, we may want to fill in the missing values in these columns or exclude the rows with missing values from the analysis. The `deathtime` column has missing values for the admissions that did not result in the patient's death, so these missing values are expected and do not necessarily need to be filled in.

Let's fill the missing values in the `language` and `religion` columns with the most common value in each column. For the `deathtime` column, we'll leave the missing values as they are, since they provide meaningful information (i.e., the patient did not die during the admission)


```python
# Fill missing values in the 'language' and 'religion' columns with the most common value
admissions_df['language'].fillna(admissions_df['language'].mode()[0], inplace=True)
admissions_df['religion'].fillna(admissions_df['religion'].mode()[0], inplace=True)

# Check for missing values again
cleaned_missing_values = admissions_df.isnull().sum()

cleaned_missing_values
```




    row_id                   0
    subject_id               0
    hadm_id                  0
    admittime                0
    dischtime                0
    deathtime               89
    admission_type           0
    admission_location       0
    discharge_location       0
    insurance                0
    language                 0
    religion                 0
    marital_status          16
    ethnicity                0
    edregtime               37
    edouttime               37
    diagnosis                0
    hospital_expire_flag     0
    has_chartevents_data     0
    stay_duration            0
    dtype: int64


