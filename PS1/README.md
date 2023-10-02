# Exercise 1
Write a function temp_tester that takes a definition of normal body temperature 
returns True if its argument is within 1 degree of normal temperature, and False if not 


```python
def temp_tester(normal_temperature):
    def is_within_one_degree(temp):
        return abs(normal_temperature - temp) <= 1

    return is_within_one_degree
```

### Test code


```python
human_tester = temp_tester(37)
chicken_tester = temp_tester(41.1)

print(chicken_tester(42)) # True -- i.e. not a fever for a chicken
print(human_tester(42))   # False -- this would be a severe fever for a human
print(chicken_tester(43)) # False
print(human_tester(35))   # False -- too low
print(human_tester(98.6)) # False -- normal in degrees F but our reference temp was in degrees C
```

    True
    False
    False
    False
    False


# Exercise 2
Quality scores are encoded such that higher scores correspond to characters with higher ASCII values.


```python
def better_quality(char1, char2):
    if ord(char1) > ord(char2):
        return char1
    elif ord(char1) < ord(char2):
        return char2
    else:
        return None  # both characters have the same quality
```


```python
# Test the function
charA = 'A'
charB = 'B'
print(better_quality(charA, charB))  # This should print 'B' since it has a higher ASCII value than 'A'
```

    B


1. Convert each character in the string to its corresponding ASCII value (numeric quality score).
2. Identify the maximum ASCII value (best quality) in the string.
3. Compute the average ASCII value.
4. Convert the average ASCII value back to its corresponding character (single-character quality score)


```python
def analyze_quality(quality_string):
    if not quality_string:
        raise ValueError("Input string is empty")

    # Convert each character to its corresponding ASCII value
    numeric_scores = [ord(char) for char in quality_string]

    # Calculate best and average quality scores
    best_quality = max(numeric_scores)
    avg_quality = sum(numeric_scores) / len(numeric_scores)

    # Convert average score to its corresponding character
    avg_char = chr(int(round(avg_quality)))

    return best_quality, avg_quality, avg_char

# Test the function
test_string = "!''*((((***+))%%%++)(%%%%).1***-+*''))**55CCF>>>>>>CCCCCCC65"
best_q, avg_q, avg_char = analyze_quality(test_string)
print(f"Best Quality: {best_q}")
print(f"Average Quality (Numeric): {avg_q}")
print(f"Average Quality (Character): {avg_char}")
```

    Best Quality: 70
    Average Quality (Numeric): 48.06666666666667
    Average Quality (Character): 0


### Choices & Discussion:

Empty String Handling: If the input string is empty, the function raises a ValueError.
Average Rounding: For the average score, I'm rounding to the nearest integer to get a valid character representation. This might slightly alter the true average, but it ensures we get a meaningful character representation.

### Testing & Convincing Explanation:

In the given test, the function will identify the character with the highest ASCII value as the best quality. The average score is computed by summing all the ASCII values and dividing by the length of the string. The average quality score as a character is derived by rounding the numeric average value and converting it back to a character.

# Exercise 3


```python
def administer_meds(delta_t, tstop):
    t = 0
    while t < tstop:
        print(f"Administering meds at t={t}")
        t += delta_t
```

## `administer_meds` Function Explanation

### Function Logic:

1. **Initialization**: The function initializes a time counter, `t`, set to 0, likely representing the starting time.
  
2. **Loop**: A `while` loop is used, which continues to execute as long as the current time `t` is less than the stopping time `tstop`.
   
3. **Administering Meds**: Within the loop, a print statement simulates the action of administering the medication at the current time `t`.

4. **Time Update**: After the simulated administration, the time `t` is incremented by `delta_t`, updating the time for the next dose.

### Relationships:

- **tstop**: This represents the total time for which the meds need to be administered.
  
- **delta_t**: Represents the time gap between two consecutive doses.
  
- **Number of doses**: The total number of doses given is approximately `tstop / delta_t` (assuming `tstop` is a multiple of `delta_t`).

For example, if the medication needs to be administered over 5 hours (`tstop=5`) with an interval of 1 hour (`delta_t=1`), then the medication will be administered 5 times in total.

### Summary:

In essence, the function simulates the periodic administration of medication at intervals of `delta_t` until the duration of `tstop` is reached. The total number of doses administered depends on the ratio of `tstop` to `delta_t`.


### Execution Results:

1. **For `administer_meds(0.25, 1)`**: 
   
   Expectation: 4 doses at times: t=0, 0.25, 0.5, and 0.75.


```python
administer_meds(0.25, 1)
```

    Administering meds at t=0
    Administering meds at t=0.25
    Administering meds at t=0.5
    Administering meds at t=0.75


2. **For `administer_meds(0.1, 1)`**: 

   Expectation: 10 doses at times: t=0, 0.1, 0.2, ... up to 0.9.


```python
administer_meds(0.1, 1)
```

    Administering meds at t=0
    Administering meds at t=0.1
    Administering meds at t=0.2
    Administering meds at t=0.30000000000000004
    Administering meds at t=0.4
    Administering meds at t=0.5
    Administering meds at t=0.6
    Administering meds at t=0.7
    Administering meds at t=0.7999999999999999
    Administering meds at t=0.8999999999999999
    Administering meds at t=0.9999999999999999


### Findings:
While the logic seems straightforward, the use of floating-point arithmetic can lead to unexpected results due to precision errors in computers. Such errors can manifest when you're repeatedly adding small decimals or when the division isn't exact in binary representation.

In the first case, you would likely get exactly what you expect. In the second case, as you approach the value of **`tstop`**, you might encounter precision issues with the floating point addition.

**Clinical Significance**:
While seemingly minor, such discrepancies in a medical setting can be very significant. If we are off by even one iteration, a patient could receive one less or one more dose than intended. Depending on the medication and condition, this could have no significant impact, or it could be potentially harmful.

### Solution Without Surprises:
One way to address this issue is by using a loop based on the number of doses rather than the continuous addition of floating point numbers:


```python
def administer_meds(delta_t, tstop):
    num_doses = int(tstop / delta_t)
    for i in range(num_doses):
        t = i * delta_t
        print(f"Administering meds at t={t}")
```

By iterating over the number of doses, you ensure that the exact number of doses is administered without the potential surprises introduced by floating-point arithmetic.

# Exercise 4

## Data Source

The COVID-19 data used in this project is sourced from [The New York Times's GitHub repository](https://github.com/nytimes/covid-19-data). We would like to thank The New York Times for making this valuable data publicly available.

## Plotting New COVID-19 Cases for Selected States

Given the dataset `us-states.csv` from The New York Times, which provides running totals of COVID-19 cases for US states, our goal is to visualize the **new cases** for a selected list of states.

### Function Design

We'll create a function `plot_new_cases` which takes a list of state names and plots the new cases against dates using overlaid line graphs.



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_new_cases(states):
    # Load the data
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
    df = pd.read_csv(url)

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(14, 8))
    
    for state in states:
        state_data = df[df['state'] == state]
        
        # Calculate the new cases
        state_data['new_cases'] = state_data['cases'].diff()
        
        # Plot new cases for the state
        plt.plot(state_data['date'], state_data['new_cases'], label=state)

    plt.title('New COVID-19 Cases vs. Date')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Limitations:

1. The first date for each state will have NaN for new cases since there's no previous data point to compare.
2. For states with similar new case trends, lines may overlap, making it difficult to differentiate.
3. The function assumes a consistent data format from The New York Times source.

### Testing the Function:

Let's visualize the new COVID-19 cases for `California`, `New York`, and `Texas`. The resulting plot shows the new cases vs. date for these states with different line colors. The legend helps in identifying which line corresponds to each state.


```python
# Testing the function
plot_new_cases(['California', 'New York', 'Texas'])
```

    <ipython-input-27-0106bf1d1eba>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      state_data['new_cases'] = state_data['cases'].diff()
    <ipython-input-27-0106bf1d1eba>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      state_data['new_cases'] = state_data['cases'].diff()
    <ipython-input-27-0106bf1d1eba>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      state_data['new_cases'] = state_data['cases'].diff()



    
![png](output_25_1.png)
    


## Date of its highest number of new cases
To achieve this, first need to process the data to compute the new cases for each state, and then identify the date with the highest number of new cases for the specified state


```python
def date_of_highest_new_cases(state_name):
    # Load the data
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
    df = pd.read_csv(url)

    # Filter data for the specified state
    state_data = df[df['state'] == state_name]
    
    # Calculate the new cases
    state_data['new_cases'] = state_data['cases'].diff()
    
    # Get the date of the highest number of new cases
    highest_date = state_data[state_data['new_cases'] == state_data['new_cases'].max()]['date'].iloc[0]
    
    return highest_date
```


```python
# Testing the function
state_to_test = "New York"
print(f"The date of the highest number of new cases for {state_to_test} is: {date_of_highest_new_cases(state_to_test)}")
```

    The date of the highest number of new cases for New York is: 2022-01-08


    <ipython-input-29-975214736e1a>:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      state_data['new_cases'] = state_data['cases'].diff()


**Note**: This function assumes that there's only one date with the highest number of new cases. If there are multiple dates with the same highest number, it'll return the first occurrence.

## Comparing Peaks of Daily New COVID-19 Cases Between Two States

We aim to compare two states to determine which one reached its peak in daily new COVID-19 cases first, and by how many days one's peak is separated from the other's.

### Function Design

We'll create a function `peak_comparison` which takes the names of two states. This function will report:

1. Which state had its highest number of daily new cases first.
2. The number of days that separate one state's peak from the other's.


```python
import pandas as pd

def peak_comparison(state1, state2):
    # Load the data
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
    df = pd.read_csv(url)

    # Function to get the date of highest new cases for a state
    def get_peak_date(state):
        state_data = df[df['state'] == state]
        state_data['new_cases'] = state_data['cases'].diff()
        return state_data[state_data['new_cases'] == state_data['new_cases'].max()]['date'].iloc[0]

    peak_date1 = pd.to_datetime(get_peak_date(state1))
    peak_date2 = pd.to_datetime(get_peak_date(state2))

    if peak_date1 < peak_date2:
        earlier_state = state1
        later_state = state2
        difference = (peak_date2 - peak_date1).days
    else:
        earlier_state = state2
        later_state = state1
        difference = (peak_date1 - peak_date2).days

    return f"{earlier_state} reached its peak number of new cases first. There are {difference} days between the peaks of {earlier_state} and {later_state}."
```


```python
# Testing the function
states_to_compare = ("California", "New York")
print(peak_comparison(*states_to_compare))
```

    New York reached its peak number of new cases first. There are 2 days between the peaks of New York and California.


    <ipython-input-34-5893a7b67187>:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      state_data['new_cases'] = state_data['cases'].diff()
    <ipython-input-34-5893a7b67187>:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      state_data['new_cases'] = state_data['cases'].diff()


This will output the state that had its peak first and the number of days separating the two peaks.

**Note**: This function assumes a unique date of the highest number of new cases for each state. If multiple dates have the same highest number for a state, the function considers the first occurrence.

# Exercise 5

## Parsing MeSH Data (`desc2023.xml`)

We aim to extract the `DescriptorName` associated with the `DescriptorUI` `D007154` from the MeSH XML data.

### Steps:
1. Download the `desc2023.xml` file from the provided URL.
2. Read the XML file using the `xml.etree.ElementTree` module in Python.
3. Traverse the XML tree to identify the `DescriptorUI` with value `D007154`.
4. Extract and display the associated `DescriptorName`.

### Python Code Implementation:


```python
import urllib.request
import xml.etree.ElementTree as ET

# Download the XML file
url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2023.xml"
urllib.request.urlretrieve(url, "desc2023.xml")
```




    ('desc2023.xml', <http.client.HTTPMessage at 0x7f9f09aaf8b0>)




```python
def get_descriptor_name(file_name, descriptor_ui):
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Traverse the XML tree to find the desired DescriptorUI
    for descriptor in root.findall('DescriptorRecord'):
        ui = descriptor.find('DescriptorUI')
        if ui is not None and ui.text == descriptor_ui:
            descriptor_name = descriptor.find('DescriptorName').find('String')
            return descriptor_name.text
    return None
```


```python
descriptor_ui_to_search = "D007154"
name = get_descriptor_name("desc2023.xml", descriptor_ui_to_search)
```


```python
if name:
    print(f"The DescriptorName associated with DescriptorUI {descriptor_ui_to_search} is: {name}")
else:
    print(f"No DescriptorName found for DescriptorUI {descriptor_ui_to_search}")
```

    The DescriptorName associated with DescriptorUI D007154 is: Immune System Diseases


## Finding DescriptorUI for a Given DescriptorName
We aim to extract the `DescriptorUI` (MeSH Unique ID) associated with the `DescriptorName` "Nervous System Diseases" from the MeSH XML data.


```python
def get_descriptor_ui(file_name, descriptor_name_target):
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Traverse the XML tree to find the desired DescriptorName
    for descriptor in root.findall('DescriptorRecord'):
        descriptor_name = descriptor.find('DescriptorName').find('String')
        if descriptor_name is not None and descriptor_name.text == descriptor_name_target:
            ui = descriptor.find('DescriptorUI')
            return ui.text
    return None
```


```python
descriptor_name_to_search = "Nervous System Diseases"
ui = get_descriptor_ui("desc2023.xml", descriptor_name_to_search)
```


```python
if ui:
    print(f"The DescriptorUI associated with DescriptorName \"{descriptor_name_to_search}\" is: {ui}")
else:
    print(f"No DescriptorUI found for DescriptorName \"{descriptor_name_to_search}\"")
```

    The DescriptorUI associated with DescriptorName "Nervous System Diseases" is: D009422


## Extracting DescriptorNames of Common Descendants in MeSH Data

Our goal is to find `DescriptorNames` in the MeSH hierarchy that are descendants of both "Nervous System Diseases" and `D007154`. The relationship between terms is determined by their `TreeNumber`, with descendants having extended `TreeNumber` values.

### Steps:
1. Extract the `TreeNumber` for both "Nervous System Diseases" and `D007154`.
2. Traverse the XML to find descendants (based on `TreeNumber`) for both terms.
3. Determine common descendants by intersecting both lists.



```python
def get_tree_numbers_for_descriptor(file_name, descriptor_ui=None, descriptor_name=None):
    tree = ET.parse(file_name)
    root = tree.getroot()
    tree_numbers = set()

    for descriptor in root.findall('DescriptorRecord'):
        ui = descriptor.find('DescriptorUI').text
        name = descriptor.find('DescriptorName').find('String').text
        
        if (descriptor_ui and descriptor_ui == ui) or (descriptor_name and descriptor_name == name):
            for tree_number_element in descriptor.findall('TreeNumberList/TreeNumber'):
                tree_numbers.add(tree_number_element.text)

    return tree_numbers
```


```python
def get_descendant_names(file_name, tree_numbers):
    tree = ET.parse(file_name)
    root = tree.getroot()
    names = set()

    for descriptor in root.findall('DescriptorRecord'):
        for tree_number_element in descriptor.findall('TreeNumberList/TreeNumber'):
            for target_tree_number in tree_numbers:
                if tree_number_element.text.startswith(target_tree_number):
                    names.add(descriptor.find('DescriptorName').find('String').text)
                    
    return names
```


```python
tree_numbers_nervous = get_tree_numbers_for_descriptor("desc2023.xml", descriptor_name="Nervous System Diseases")
tree_numbers_d007154 = get_tree_numbers_for_descriptor("desc2023.xml", descriptor_ui="D007154")
print(tree_numbers_nervous)
print(tree_numbers_d007154)
```

    {'C10'}
    {'C20'}


The MeSH tree number of "Nervous System Diseases" and D007154 are “C10” and “C20” respectively.


```python
descendant_names_nervous = get_descendant_names("desc2023.xml", tree_numbers_nervous)
descendant_names_d007154 = get_descendant_names("desc2023.xml", tree_numbers_d007154)
```


```python
# Find intersection of the two sets to get common descendants
common_descendants = descendant_names_nervous.intersection(descendant_names_d007154)

print(common_descendants)
```

    {'Multiple Sclerosis', 'Autoimmune Diseases of the Nervous System', 'Multiple Sclerosis, Relapsing-Remitting', 'Anti-N-Methyl-D-Aspartate Receptor Encephalitis', 'AIDS Dementia Complex', 'Giant Cell Arteritis', 'Encephalomyelitis, Acute Disseminated', 'Myelitis, Transverse', 'Multiple Sclerosis, Chronic Progressive', 'Encephalomyelitis, Autoimmune, Experimental', 'Demyelinating Autoimmune Diseases, CNS', 'AIDS Arteritis, Central Nervous System', 'Myasthenia Gravis, Neonatal', 'Myasthenia Gravis', 'Myasthenia Gravis, Autoimmune, Experimental', 'Nervous System Autoimmune Disease, Experimental', 'Lambert-Eaton Myasthenic Syndrome', 'POEMS Syndrome', 'Uveomeningoencephalitic Syndrome', 'Leukoencephalitis, Acute Hemorrhagic', 'Kernicterus', 'Polyradiculoneuropathy', 'Ataxia Telangiectasia', 'Guillain-Barre Syndrome', 'Vasculitis, Central Nervous System', 'Diffuse Cerebral Sclerosis of Schilder', 'Microscopic Polyangiitis', 'Autoimmune Hypophysitis', 'Mevalonate Kinase Deficiency', 'Stiff-Person Syndrome', 'Miller Fisher Syndrome', 'Neuromyelitis Optica', 'Lupus Vasculitis, Central Nervous System', 'Polyradiculoneuropathy, Chronic Inflammatory Demyelinating', 'Neuritis, Autoimmune, Experimental'}


## Retrieved Results Overview

The results obtained point towards conditions that span the intricacies of both the **immune** and **nervous systems**. These conditions can be categorized as follows:

- **Autoimmune Disorders Affecting the Nervous System**: Conditions where the body's immune response mistakenly targets and damages its own nervous system. Examples include:
  - Multiple Sclerosis
  - Miller Fisher Syndrome

- **Nervous System Inflammatory Conditions**: Diseases characterized by inflammation predominantly within the nervous system. An example is:
  - Transverse Myelitis

- **Immune Hemolytic Diseases Leading to Neurological Impairment**: Diseases where an immune response against certain blood components causes neurological issues. An example is:
  - Kernicterus

- **Hereditary Disorders Impacting Both Systems**: These are genetically inherited conditions that manifest symptoms in both the immune and nervous systems. For instance:
  - Ataxia Telangiectasia

- **Neurological Manifestations from Severe Infections**: Some infections, when severe, can have pronounced neurological symptoms. A notable example is:
  - AIDS Dementia Complex: This results from an advanced HIV infection primarily affecting the brain.
