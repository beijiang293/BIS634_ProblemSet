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


```python
# Test
administer_meds(0.1, 1)
```

    Administering meds at t=0.0
    Administering meds at t=0.1
    Administering meds at t=0.2
    Administering meds at t=0.30000000000000004
    Administering meds at t=0.4
    Administering meds at t=0.5
    Administering meds at t=0.6000000000000001
    Administering meds at t=0.7000000000000001
    Administering meds at t=0.8
    Administering meds at t=0.9


# Exercise 4


```python

```
