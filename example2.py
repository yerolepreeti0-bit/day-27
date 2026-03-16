# Pizza Price Predictor using Linear Regression

# Step 1: Import library
from sklearn.linear_model import LinearRegression

# Step 2: Training Data
# X = [Diameter, Toppings]
X = [
    [8, 1],
    [8, 3],
    [12, 1],
    [12, 4],
    [16, 2]
]

# Y = Pizza Price
y = [10, 13, 18, 22.5, 28]

# Step 3: Create Model
model = LinearRegression()

# Step 4: Train the model
model.fit(X, y)

# Step 5: Predict pizza price
diameter = 10
toppings = 2

prediction = model.predict([[diameter, toppings]])

print("Predicted Pizza Price:", prediction[0])