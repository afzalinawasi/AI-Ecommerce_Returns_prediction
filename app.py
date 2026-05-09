import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Ecommerce Return Predictor",
    page_icon="🛒",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛒 Ecommerce Return Prediction App")

st.markdown("""
This AI-powered app predicts whether an ecommerce order is likely to be returned.

It uses order details such as product category, price, quantity, region, payment method, 
delivery days, customer rating, discount, and order timing to estimate return risk.
""")

with st.expander("📌 When to use this predictor"):
    st.write("""
    Use this app before dispatching an ecommerce order to estimate its return risk.

    It can help ecommerce teams identify orders that may need extra review, clearer product communication, 
    customer confirmation, or better delivery follow-up.

    The goal is not to block an order, but to support smarter return-prevention decisions.
    """)

with st.expander("⚠️ How to interpret the outcome"):
    st.write("""
    - This app is a learning project built using a synthetic ecommerce dataset.
    - The prediction should be treated as a return-risk signal, not a final business decision.
    - A high probability means the order may need extra attention before dispatch.
    - A low probability does not guarantee that the order will not be returned.
    - This app cannot replace real ecommerce policies, customer support judgment, or production-grade analytics.
    """)


# Load saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load preprocessing data
with open("preprocessing.pkl", "rb") as file:
    preprocessing_data = pickle.load(file)


st.header("Enter Order Details")

product_category = st.selectbox(
    "Product Category",
    ["Automotive", "Beauty", "Electronics", "Fashion", "Home", "Sports", "Toys"]
)

region = st.selectbox(
    "Region",
    ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["BankTransfer", "Cash", "CreditCard", "PayPal"]
)

product_price = st.number_input("Product Price", min_value=0.0, max_value=1000.0, value=250.0)
quantity = st.number_input("Quantity", min_value=1, max_value=20, value=3)
delivery_days = st.number_input("Delivery Days", min_value=1, max_value=30, value=5)
customer_rating = st.slider("Customer Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
discount_percent = st.slider("Discount Percent", min_value=0, max_value=100, value=5)

order_year = st.selectbox("Order Year", [2023, 2024, 2025])
order_month = st.selectbox("Order Month", list(range(1, 13)))
order_day_of_week = st.selectbox(
    "Order Day of Week",
    [0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
)

input_data = pd.DataFrame({
    "product_category": [product_category],
    "product_price": [product_price],
    "quantity": [quantity],
    "region": [region],
    "payment_method": [payment_method],
    "delivery_days": [delivery_days],
    "customer_rating": [customer_rating],
    "discount_percent": [discount_percent],
    "order_year": [order_year],
    "order_month": [order_month],
    "order_day_of_week": [order_day_of_week]
})

st.subheader("Input Preview")
st.dataframe(input_data)

scaler = preprocessing_data["scaler"]
feature_columns = preprocessing_data["feature_columns"]
categorical_features = preprocessing_data["categorical_features"]
numerical_features = preprocessing_data["numerical_features"]

if st.button("Predict Return"):
    input_scaled = input_data.copy()

    input_scaled[numerical_features] = scaler.transform(
        input_scaled[numerical_features]
    )

    input_encoded = pd.get_dummies(
    input_scaled,
    columns=categorical_features,
    drop_first=False
    )   

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")

    if probability >= 0.50:
        prediction_label = "Likely to be Returned"
        risk_level = "High"
        recommendation = (
            f"This order has high return risk. Since the product category is {product_category}, "
            "review product description, sizing/specifications, product images, delivery promise, "
            "and return-policy visibility. Consider adding extra confirmation before checkout or "
            "offering support before dispatch."
        )
        st.error(f"Prediction: {prediction_label}")

    elif probability >= 0.30:
        prediction_label = "Not Likely to be Returned"
        risk_level = "Medium"
        recommendation = (
            f"This order has moderate return risk. For {product_category}, make sure the customer has "
            "clear product details, accurate expectations, and delivery information. A small reminder "
            "or confirmation message can help reduce avoidable returns."
        )
        st.warning(f"Prediction: {prediction_label}")

    else:
        prediction_label = "Not Likely to be Returned"
        risk_level = "Low"
        recommendation = (
            f"This order has low return risk. Continue maintaining clear product listings, reliable "
            f"delivery, and good customer experience for {product_category} orders."
        )
        st.success(f"Prediction: {prediction_label}")

    st.metric("Return Probability", f"{probability:.2%}")
    st.progress(float(probability))

    st.info(f"Risk level: {risk_level}")

    st.subheader("💡 Business Recommendation")
    st.info(recommendation)
