
## R1: Avoid Hard-Coded Numbers

Don't use mysterious numbers in your formulas. Give them a name so others know what they represent.

* **Bad Code:**

 ```python
 def calculate_total(price):
 # What is 0.08? A tax? A fee? A discount?
 return price + (price * 0.08)
 ```

* **Clean Code:**

 ```python
 def calculate_total(price):
 SALES_TAX_RATE = 0.08
 return price + (price * SALES_TAX_RATE)
 ```

## R2: Use Meaningful and Descriptive Names

Names should explain the variable's purpose without needing a comment.

* **Bad Code:**

 ```python
 # Ambiguous names
 d = 12 # elapsed time in days
 def get_m():
  pass
 ```

* **Clean Code:**

 ```python
 elapsed_days = 12
 def get_active_members():
  pass
 ```

## R3: Use Comments Sparingly

Code should be clear enough to read without comments. Only comment to explain *why* something strange is happening.

* **Bad Code:**

 ```python
 # Check if age is greater than 18
 if age > 18:
  # Set status to adult
  status = "adult"
 ```

* **Clean Code:**

 ```python
 if age > 18:
  status = "adult"
 ```

## R4: Write Short Functions (Single Responsibility)

A function should do one thing only. If it does "X *and* Y," split it up.

* **Bad Code:**

 ```python
 def process_user(user):
  # Validates user
  if not user.email:
   return False
  # Saves to database
  db.save(user)
  # Sends email
  email_service.send_welcome(user)
 ```

* **Clean Code:**

 ```python
 def register_user(user):
  if is_valid_user(user):
   save_user_to_db(user)
   send_welcome_email(user)
 
 def is_valid_user(user):
  return bool(user.email)
 ```

### Clean Code in Jupyter Notebooks

**Focus each cell on one task**

* Each cell should perform a single, clear task. If a cell is doing multiple things (e.g., data loading, data cleaning, and plotting), **split it up** into separate cells. This improves readability and makes it easier to debug and maintain the code.
* **Example**: Instead of doing everything in a single cell, have separate cells for:
 1. **Loading data**
 2. **Data preprocessing**
 3. **Data visualization**

## R5: Document Complex Interfaces (Contracts)

For complex functions or classes, use structured documentation (Docstrings) to explain the **Inputs (Args)**, **Outputs (Returns)**, and **Errors (Raises)**. Create a clear and concise "contract" so the user doesn't need to read the implementation logic.

* **Bad Code:**

 ```python
 # Logic is hidden, inputs are vague, return type is unknown
 def calc_loan(p, r, y):
  n = y * 12
  r = r / 100 / 12
  return p * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
 ```

* **Clean Code:**

 ```python
 def calculate_amortization(principal, annual_rate, years):
  """
  Calculates the monthly payment for a fixed-rate loan.
 
  Args:
   principal (float): The total amount borrowed.
   annual_rate (float): The annual interest rate as a percentage (e.g., 5.0).
   years (int): The term of the loan in years.
 
  Returns:
   float: The fixed monthly payment amount.
 
  Raises:
   ValueError: If years is 0 or principal is negative.
  """
  if years <= 0:
    raise ValueError("Loan term must be positive.")
  
  # ... calculation logic ...
  return monthly_payment
 ```

## R6: Follow the DRY (Don't Repeat Yourself) Principle

If you see the same logic twice, move it into a shared function.

* **Bad Code:**

 ```python
 def area_rectangle(w, h):
  return w * h
 
 def area_square(s):
  return s * s # Repeated logic (width * height)
 ```

* **Clean Code:**

 ```python
 def calculate_area(width, height):
  return width * height
 
 def area_square(side):
  return calculate_area(side, side)
 ```

## R7: Follow Established Standards

Follow the style guide for your language (e.g., Python uses `snake_case`, Java uses `camelCase`).

* **Bad Code (Python):**

 ```python
 # Python convention is snake_case, not camelCase
 def CalculateTotalPrice(userInput):
  pass
 ```

* **Clean Code (Python):**

 ```python
 def calculate_total_price(user_input):
  pass
 ```

## R8: Encapsulate Nested Conditionals

Deeply nested `if/else` blocks are hard to read. Move the logic into its own function.

* **Bad Code:**

 ```python
 if user.is_active:
  if user.has_subscription:
   if user.credits > 10:
    grant_access()
 ```

* **Clean Code:**

 ```python
 if should_grant_access(user):
  grant_access()
 
 def should_grant_access(user):
  return user.is_active and user.has_subscription and user.credits > 10
 ```

## R9: Refactor Continuously

Don't wait for a "cleanup phase." Improve code as you touch it.

* **Example Scenario:**
 	* *Bad approach:* You see a messy function but think, "I'll fix it next month when we have time." (You never will).
 	* *Clean approach:* You see a messy function while fixing a bug. You spend 5 extra minutes renaming variables and simplifying loops before submitting your fix.

## R10: Use Version Control

Commit small chunks of work often rather than one massive file at the end of the week.

* **Example Scenario:**
 	* *Bad approach:* Saving files as `script_final.py`, `script_final_v2.py`, `script_really_final.py`.
 	* *Clean approach:* Using Git to commit changes with messages like "Fix login bug" and "Refactor user validation logic."
