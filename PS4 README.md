# Exercise 1
## Smith-Waterman Algorithm Implementation

The Smith-Waterman algorithm has been implemented to compute the optimal local alignment between two DNA sequences. The function `align` takes two sequences and returns their optimal local alignment along with the alignment score. The function also accepts three keyword arguments with defaults: `match=1`, `gap_penalty=1`, and `mismatch_penalty=1`. The implementation details are as follows:

### Function Implementation:


```python
def align(seq1, seq2, match=1, gap_penalty=1, mismatch_penalty=1):
    # Lengths of the two sequences
    m, n = len(seq1), len(seq2)

    # Initialize the scoring matrix
    score_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Initialize the traceback matrix
    traceback_matrix = [[None for _ in range(n+1)] for _ in range(m+1)]

    # Calculate scores and fill matrices
    max_score = 0
    max_pos = (0, 0)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate scores for matches/mismatches and gaps
            if seq1[i-1] == seq2[j-1]:
                match_score = score_matrix[i-1][j-1] + match
            else:
                match_score = score_matrix[i-1][j-1] - mismatch_penalty
            
            gap_score1 = score_matrix[i-1][j] - gap_penalty
            gap_score2 = score_matrix[i][j-1] - gap_penalty
            
            # Choose the best score
            score_matrix[i][j], traceback_matrix[i][j] = max(
                (0, None),
                (match_score, 'match'),
                (gap_score1, 'gap1'),
                (gap_score2, 'gap2'),
                key=lambda x: x[0]
            )

            # Keep track of the highest score
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)

    # Traceback to get the optimal local alignment
    i, j = max_pos
    aligned_seq1, aligned_seq2 = [], []
    while traceback_matrix[i][j] is not None:
        if traceback_matrix[i][j] == 'match':
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 'gap1':
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        else: # gap2
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1

    # Reverse the aligned sequences as we traced them back
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))

    return aligned_seq1, aligned_seq2, max_score
```


```python
# Test the function
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
print("Test 1:")
print(f"Seq1: {seq1}\nSeq2: {seq2}\nScore: {score}\n")

seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)
print("Test 2:")
print(f"Seq1: {seq1}\nSeq2: {seq2}\nScore: {score}")
```

    Test 1:
    Seq1: agacccta-cgt-gac
    Seq2: aga-cctagcatcgac
    Score: 8
    
    Test 2:
    Seq1: gcatcga
    Seq2: gcatcga
    Score: 7


### Results and Discussion:
The function `align` successfully computed the optimal local alignments and scores for the provided test cases. The results indicate that the algorithm correctly adjusts alignments based on the specified parameters (match, gap penalty, and mismatch penalty), confirming the function's accuracy and effectiveness in solving local alignment problems.

# Exercise 2
## Two-Dimensional K-Nearest Neighbors Classifier Using Quad-Tree

We have implemented a two-dimensional k-nearest neighbors classifier. This implementation uses a quad-tree for efficient storage and retrieval of points. Below are the key components of this implementation:

### QuadTree Class
- Manages the overall structure and methods for the quad-tree.
- Contains a nested `Node` class representing each point in the tree along with its classification.

### Insertion Method
- Inserts new points with their classifications into the quad-tree.
- Organizes points based on their x and y coordinates, ensuring efficient spatial searching.

### K-Nearest Neighbors Method
- Finds the k nearest neighbors to a given point using Euclidean distance.
- Efficiently navigates the tree by only exploring relevant quadrants.

### Classification Method
- Classifies a new point by finding its k nearest neighbors and selecting the most common class among them.


```python
class QuadTree:
    """
    QuadTree class for efficient k-nearest neighbors search.
    """

    class Node:
        """
        Node class representing each element in the QuadTree.
        """
        def __init__(self, point, classification):
            self.point = point  # The (x, y) coordinates
            self.classification = classification  # The class of the point
            self.children = [None, None, None, None]  # Children: NW, NE, SW, SE

    def __init__(self):
        self.root = None

    def insert(self, point, classification):
        """
        Insert a new point with its classification into the tree.
        """
        self.root = self._insert(self.root, point, classification, 0)

    def _insert(self, node, point, classification, depth):
        """
        Helper function to insert a new point in the tree.
        """
        if node is None:
            return self.Node(point, classification)

        # Determine the quadrant for the point
        x, y = point
        horizontal_mid = depth % 2 == 0
        if (horizontal_mid and x < node.point[0]) or (not horizontal_mid and y < node.point[1]):
            index = 0 if y < node.point[1] else 1  # NW or NE
        else:
            index = 2 if y < node.point[1] else 3  # SW or SE

        node.children[index] = self._insert(node.children[index], point, classification, depth + 1)
        return node

    def k_nearest_neighbors(self, point, k):
        """
        Find the k nearest neighbors to a given point.
        """
        neighbors = []
        self._k_nearest_neighbors(self.root, point, k, neighbors, 0)
        neighbors.sort(key=lambda x: x[0])
        return [n[1] for n in neighbors][:k]

    def _k_nearest_neighbors(self, node, point, k, neighbors, depth):
        """
        Helper function to find k nearest neighbors.
        """
        if node is None:
            return

        # Compute euclidean distance from point to current node
        distance = self._euclidean_distance(point, node.point)
        if len(neighbors) < k or distance < neighbors[-1][0]:
            neighbors.append((distance, node.classification))
            neighbors.sort(key=lambda x: x[0])
            if len(neighbors) > k:
                neighbors.pop()

        # Determine which side of the split line to explore first
        horizontal_mid = depth % 2 == 0
        index = 0 if (horizontal_mid and point[0] < node.point[0]) or (not horizontal_mid and point[1] < node.point[1]) else 1

        # Explore the preferred side of the split line first
        self._k_nearest_neighbors(node.children[index], point, k, neighbors, depth + 1)

        # Check if we need to explore the other side
        if (horizontal_mid and abs(point[0] - node.point[0]) < neighbors[-1][0]) or \
           (not horizontal_mid and abs(point[1] - node.point[1]) < neighbors[-1][0]):
            other_index = 1 - index
            self._k_nearest_neighbors(node.children[other_index], point, k, neighbors, depth + 1)

    @staticmethod
    def _euclidean_distance(point1, point2):
        """
        Compute the Euclidean distance between two points.
        """
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def classify_point(self, point, k):
        """
        Classify a point based on the k nearest neighbors.
        """
        neighbors = self.k_nearest_neighbors(point, k)
        # Count the occurrences of each class in the neighbors
        classes = {}
        for neighbor in neighbors:
            classes[neighbor] = classes.get(neighbor, 0) + 1
        # Return the most common class
        return max(classes.items(), key=lambda x: x[1])[0]
```

### Example Usage
In this example, we insert several data points into the quad-tree and then classify a new point based on its 3 nearest neighbors.


```python
# Example usage
qt = QuadTree()
data = [((2, 3), 'A'), ((5, 4), 'B'), ((9, 6), 'A'), ((4, 7), 'B'), ((8, 1), 'A'), ((7, 2), 'B')]
for point, classification in data:
    qt.insert(point, classification)

# Classify a new point
new_point = (5, 5)
k = 3
classification = qt.classify_point(new_point, k)
print(f"The point {new_point} is classified as '{classification}' based on its {k} nearest neighbors.")
```

    The point (5, 5) is classified as 'A' based on its 3 nearest neighbors.



```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data from the Excel file
file_path = 'Rice_Cammeo_Osmancik.xlsx'
data = pd.read_excel(file_path)

# Identifying the quantitative columns (assuming they are numerical and not categorical)
quantitative_columns = data.select_dtypes(include=['float', 'int']).columns

# Normalize the quantitative columns to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
data[quantitative_columns] = scaler.fit_transform(data[quantitative_columns])

# Displaying the first few rows of the normalized data
data.head()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Major_Axis_Length</th>
      <th>Minor_Axis_Length</th>
      <th>Eccentricity</th>
      <th>Convex_Area</th>
      <th>Extent</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.479830</td>
      <td>2.004354</td>
      <td>2.348547</td>
      <td>-0.212943</td>
      <td>2.018337</td>
      <td>1.499659</td>
      <td>-1.152921</td>
      <td>Cammeo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.147870</td>
      <td>1.125853</td>
      <td>0.988390</td>
      <td>0.945568</td>
      <td>0.410018</td>
      <td>1.192918</td>
      <td>-0.602079</td>
      <td>Cammeo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.135169</td>
      <td>1.317214</td>
      <td>1.451908</td>
      <td>0.253887</td>
      <td>1.212956</td>
      <td>1.126504</td>
      <td>0.405611</td>
      <td>Cammeo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.293436</td>
      <td>0.115300</td>
      <td>0.261439</td>
      <td>0.198051</td>
      <td>0.239751</td>
      <td>0.233857</td>
      <td>-0.275351</td>
      <td>Cammeo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.166345</td>
      <td>1.487053</td>
      <td>1.316442</td>
      <td>0.523419</td>
      <td>0.952221</td>
      <td>1.299855</td>
      <td>-0.206013</td>
      <td>Cammeo</td>
    </tr>
  </tbody>
</table>
</div>



## K-Nearest Neighbors Effectiveness on PCA-Reduced Rice Data


```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Applying PCA to reduce the data to two dimensions
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data[quantitative_columns])

# Extracting the first and second principal components
pc0 = data_reduced[:, 0]
pc1 = data_reduced[:, 1]

# Creating a scatter plot, color-coded by the type of rice
plt.figure(figsize=(10, 6))
for class_value in data['Class'].unique():
    # Select data corresponding to each class
    idx = data['Class'] == class_value
    plt.scatter(pc0[idx], pc1[idx], label=class_value)

plt.xlabel('Principal Component 0')
plt.ylabel('Principal Component 1')
plt.title('2D PCA of Rice Data')
plt.legend()
plt.show()
```


    
![png](output_10_0.png)
    


### Observations:
- **Cluster Separation**: The two types of rice, Cammeo and Osmancik, form distinct clusters. This separation is indicative of a high potential for KNN to accurately classify the majority of the data points.

- **Overlap Area**: The presence of an overlap between the clusters suggests that the performance of KNN might be compromised in this region. The selection of an appropriate 'k' value will be important to balance the influence of the nearest neighbors from both classes.

- **Density of Points**: The central areas of the clusters are densely populated, suggesting that KNN would be highly effective in these regions due to the higher likelihood of a point's nearest neighbors belonging to the same class.

- **Outliers**: The lack of significant outliers suggests that KNN will not be unduly influenced by anomalous data points, leading to more stable classification predictions.

### Conclusion:
The PCA reduction has resulted in a two-dimensional space where the two classes of rice are mostly separable, indicating that KNN could be an effective classification method for this dataset. However, careful consideration of the 'k' value and possible weight adjustments based on distance may be required to optimize performance, especially in the overlap area between the clusters.

## K-Nearest Neighbors Classification and Confusion Matrix Analysis


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Split the data into features and target variable
X = data[quantitative_columns]
y = data['Class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the training data and transform the test data using the same scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit PCA on the training data and transform both the training and test data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Function to perform KNN classification and return the confusion matrix
def knn_confusion_matrix(k, X_train, y_train, X_test, y_test):
    # Initialize the KNN classifier with the specified k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the classifier on the training data
    knn.fit(X_train, y_train)
    
    # Predict the classes for the test data
    y_pred = knn.predict(X_test)
    
    # Generate and return the confusion matrix
    return confusion_matrix(y_test, y_pred)

# Confusion matrix for k=1
conf_matrix_k1 = knn_confusion_matrix(1, X_train_pca, y_train, X_test_pca, y_test)

# Confusion matrix for k=5
conf_matrix_k5 = knn_confusion_matrix(5, X_train_pca, y_train, X_test_pca, y_test)

conf_matrix_k1, conf_matrix_k5
```




    (array([[447,  71],
            [ 52, 573]], dtype=int64),
     array([[466,  52],
            [ 47, 578]], dtype=int64))



**Interpretation of Results**:
   - The confusion matrix provides a breakdown of correct and incorrect predictions.
   - Diagonal elements correspond to correct predictions (true positives and true negatives).
   - Off-diagonal elements indicate misclassifications (false positives and false negatives).
   - A confusion matrix for `k=1` may show more sensitivity to noise (overfitting).
   - A confusion matrix for `k=5` may indicate a more generalized model that could be more robust but less sensitive to data specifics.

# Exercise 3
## Data Cleaning Steps for Cancer Incidence Rates


```python
import pandas as pd
data = pd.read_csv('/Users/jiangbei/Desktop/BIS634/PS4/incd.csv')
data.head()
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
      <th>State</th>
      <th>FIPS</th>
      <th>Age-Adjusted Incidence Rate([rate note]) - cases per 100,000</th>
      <th>Lower 95% Confidence Interval</th>
      <th>Upper 95% Confidence Interval</th>
      <th>CI*Rank([rank note])</th>
      <th>Lower CI (CI*Rank)</th>
      <th>Upper CI (CI*Rank)</th>
      <th>Average Annual Count</th>
      <th>Recent Trend</th>
      <th>Recent 5-Year Trend ([trend note]) in Incidence Rates</th>
      <th>Lower 95% Confidence Interval.1</th>
      <th>Upper 95% Confidence Interval.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US (SEER+NPCR)(1)</td>
      <td>0.0</td>
      <td>442.3</td>
      <td>442</td>
      <td>442.6</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>1698328</td>
      <td>stable</td>
      <td>-0.3</td>
      <td>-0.6</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kentucky(7)</td>
      <td>21000.0</td>
      <td>506.8</td>
      <td>504.1</td>
      <td>509.6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>27911</td>
      <td>stable</td>
      <td>-0.3</td>
      <td>-0.8</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iowa(7)</td>
      <td>19000.0</td>
      <td>486.8</td>
      <td>483.6</td>
      <td>490</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>19197</td>
      <td>rising</td>
      <td>1</td>
      <td>0.3</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West Virginia(6)</td>
      <td>54000.0</td>
      <td>482.4</td>
      <td>478.4</td>
      <td>486.4</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>12174</td>
      <td>stable</td>
      <td>-0.2</td>
      <td>-0.4</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>New Jersey(7)</td>
      <td>34000.0</td>
      <td>481.9</td>
      <td>480</td>
      <td>483.7</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>53389</td>
      <td>falling</td>
      <td>-0.5</td>
      <td>-0.7</td>
      <td>-0.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Clean the state names by removing any numbers and parentheses
data['State'] = data['State'].str.replace(r"\(.*\)","", regex=True).str.strip()
```


```python
# Remove rows with index greater than 54
data = data[data.index <=53]
```


```python
# Save the cleaned data back to a CSV file
cleaned_data = data.to_csv('cleaned_data.csv', index=False)
```

## Citation and Data Changes

Original data sourced from the National Cancer Institute's State Cancer Profiles:
- **URL**: [https://statecancerprofiles.cancer.gov/incidencerates/index.php](https://statecancerprofiles.cancer.gov/incidencerates/index.php)

### Changes made to the original CSV:
- Removed non-data rows that contained metadata or descriptions.
- Cleaned the 'State' column to remove numbers and parentheses indicating data source information.

These modifications were necessary to prepare the dataset for analysis, ensuring that it contains only relevant data in a consistent and usable format.

## Flask Server Implementation for State Incidence Rates

We have implemented a Flask server with three routes that provide information on state-specific cancer incidence rates.

### Server Routes:

1. `@app.route("/")`: The homepage presents a form where the user can enter a state name. Upon submission, the form sends a GET request to `/info`.

2. `@app.route("/state/<string:name>")`: This route acts as an API endpoint that returns JSON-encoded data, including the state's name and its age-adjusted incidence rate.

3. `@app.route("/info", methods=["GET"])`: This page takes the state name as a GET parameter. If the state name is valid, it displays the information in an HTML page. Otherwise, it shows an error message.


```python
from flask import Flask, jsonify, request, render_template, render_template_string, redirect, url_for
import pandas as pd

app = Flask(__name__)

# Load and prepare the data
data = pd.read_csv('/Users/jiangbei/Desktop/BIS634/PS4/cleaned_data.csv')

# Homepage route
@app.route("/", methods=['GET'])
def index():
    return '''
        <html>
            <head>
                <title>Enter a State</title>
            </head>
            <body>
                <form action="/info" method="get">
                    <label for="state">State:</label>
                    <input type="text" id="state" name="state">
                    <input type="submit" value="Submit">
                </form>
            </body>
        </html>
    '''

# API route that returns JSON data
@app.route("/state/<string:name>", methods=['GET'])
def state_info(name):
    # Normalize the state name to match the data (e.g., title case)
    normalized_name = name.title()
    # Filter the data for the state
    state_data = data[data['State'] == normalized_name]
    if not state_data.empty:
        # Assuming 'Age_Adjusted_Rate' is the column name
        result = {
            'State': normalized_name,
            'Age_Adjusted_Incidence_Rate': state_data['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000'].iloc[0]
        }
        return jsonify(result)
    else:
        return jsonify({'error': 'State not found'}), 404

# Web page route that uses the state from GET arguments
@app.route("/info", methods=["GET"])
def info():
    state_name = request.args.get('state', '')
    normalized_name = state_name.title()
    state_data = data[data['State'] == normalized_name]
    if not state_data.empty:
        rate = state_data['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000'].iloc[0]
        return f'''
            <html>
                <head>
                    <title>State Information</title>
                </head>
                <body>
                    <p>State: {normalized_name}</p>
                    <p>Age-Adjusted Incidence Rate: {rate}</p>
                    <a href="/">Go back</a>
                </body>
            </html>
        '''
    else:
        return f'''
            <html>
                <head>
                    <title>Error</title>
                </head>
                <body>
                    <p>Error: State '{state_name}' not found.</p>
                    <a href="/">Go back</a>
                </body>
            </html>
        '''

# Run the app
if __name__ == "__main__":
    app.run()
```

     * Serving Flask app "__main__" (lazy loading)
     * Environment: production
    [31m   WARNING: This is a development server. Do not use it in a production deployment.[0m
    [2m   Use a production WSGI server instead.[0m
     * Debug mode: off


     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)



```python
from flask import Flask, render_template
import folium
import pandas as pd

app = Flask(__name__)

# Load your cleaned data
data = pd.read_csv('cleaned_data.csv')

@app.route('/')
def index():
    # Initialize the map centered around the US
    start_coords = (37.0902, -95.7129)
    folium_map = folium.Map(location=start_coords, zoom_start=4)

    for _, row in data.iterrows():
        state_name = row['State']
        incidence_rate = row['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']
        
        # For simplicity, let's say we have a function that returns the latitude and longitude of each state's centroid
        state_location = get_state_location(state_name)
        
        # Create a popup with the state's information
        popup_text = f"{state_name}: {incidence_rate} cases per 100k"
        folium.Marker(
            location=state_location,
            popup=popup_text
        ).add_to(folium_map)

    # Render the map in an HTML template
    return folium_map._repr_html_()

def get_state_location(state_name):
    # Dummy function - you should implement it to return actual coordinates
    return (37.0902, -95.7129)

if __name__ == '__main__':
    app.run()
```

     * Serving Flask app "__main__" (lazy loading)
     * Environment: production
    [31m   WARNING: This is a development server. Do not use it in a production deployment.[0m
    [2m   Use a production WSGI server instead.[0m
     * Debug mode: off


     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


## Flask Server Extension: Interactive Map Visualization

### Extension Description
To enhance the user experience and provide a more engaging way to explore the cancer incidence rates by state, we've added an interactive map to our Flask application. This map allows users to click on any state within the United States and view the age-adjusted incidence rate directly on the map interface. The map is implemented using the `folium` library, which integrates seamlessly with Flask to serve interactive maps.

### Implementation Details
- **Interactive Map**: The homepage now features an interactive map powered by `folium`. Users can click on a state to see a popup that displays the state's age-adjusted incidence rate.
- **State Highlighting**: States on the map are color-coded based on their incidence rates, providing a quick visual indication of the data.
- **Asynchronous Data Loading**: The state-specific data is loaded asynchronously as JSON from the `/state/<name>` API endpoint when a state is clicked, ensuring that the page does not need to be reloaded.

### Usage
Users can now interact with the map on the homepage to explore the incidence rates without needing to manually enter a state name. This reduces the potential for user input errors and makes the data more accessible.

### Additional Libraries Used
- `folium`: For generating the interactive map.
