
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
from datetime import date
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from database import create_database, save_food, get_food_history


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FoodAI | Smart Food Recognition",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

create_database()


# ==========================================
# CUSTOM FOOD THEME
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #FFF9F0;
}

[data-testid="stHeader"] {
    background: rgba(255,249,240,0.9);
}

[data-testid="stSidebar"] {
    background: #3B2419;
}

[data-testid="stSidebar"] * {
    color: #FFF8EE !important;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #3B2419;
    margin-bottom: 0;
}

.subtitle {
    color: #856B5B;
    font-size: 16px;
    margin-bottom: 20px;
}

.hero {
    background: linear-gradient(120deg, #F47B35, #E95D27);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(180, 85, 30, 0.15);
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #FFF2E7;
    font-size: 15px;
    margin-bottom: 0;
}

.section-heading {
    color: #3B2419;
    font-size: 24px;
    font-weight: 700;
    margin-top: 15px;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #F2E4D6;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
}

.food-result {
    background: #FFF0E5;
    border-left: 5px solid #F47B35;
    padding: 20px;
    border-radius: 14px;
    margin: 15px 0;
}

.food-result h2 {
    color: #3B2419;
    margin: 0;
}

.small-muted {
    color: #856B5B;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #F0E2D4;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(70,40,20,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #856B5B;
}

div[data-testid="stMetricValue"] {
    color: #E96A2C;
    font-weight: 700;
}

.stButton > button {
    background: #F47B35;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #D95720;
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #F0B58E;
    border-radius: 15px;
    padding: 15px;
}

.stSelectbox, .stNumberInput {
    background: #FFFFFF;
}

hr {
    border-color: #F0E2D4;
}

.footer {
    text-align: center;
    color: #927968;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🍽️ FoodAI")
    st.caption("SMART FOOD RECOGNITION")
    st.divider()

    st.markdown("### 🧠 AI Food Scanner")
    st.write("Identify Indian and popular food using artificial intelligence.")

    st.markdown("### ✨ Features")
    st.write("📸 Food image scanning")
    st.write("🤖 AI food prediction")
    st.write("🥗 Nutrition estimation")
    st.write("📊 Daily nutrition tracking")
    st.write("💾 Food history")

    st.divider()

    st.markdown("### 🍛 Food Categories")
    st.metric("Available Categories", "43")

    with st.expander("View all food categories"):
        st.write("""
        Aloo Gobi, Aloo Masala, Bhindi Masala,
        Biryani, Burger, Butter Naan, Chai,
        Chapati, Chicken Curry, Chole,
        Chole Bhature, Coconut Chutney,
        Dal Makhani, Dhokla, Dum Aloo,
        Fish Curry, Fried Rice, Ghevar,
        Green Chutney, Gulab Jamun, Idli,
        Jalebi, Kaathi Rolls, Kadai Paneer,
        Kebab, Kheer, Kulfi, Lassi,
        Masala Dosa, Modak, Momos,
        Mutton Curry, Paani Puri, Pakode,
        Palak Paneer, Pav Bhaji, Pizza,
        Poha, Rajma Curry, Ras Malai,
        Samosa, Shahi Paneer, White Rice.
        """)

    st.divider()
    st.caption("Powered by TensorFlow + Streamlit")


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🍲 Welcome to FoodAI</h1>
    <p>Your smart food companion. Scan food, discover what is on your plate,
    and explore its estimated nutritional value.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p class='main-title'>Discover What's on Your Plate</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Upload a food photo or use your camera to let AI identify it.</p>",
    unsafe_allow_html=True
)


# ==========================================
# LOAD AI MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/food_model_43.keras",
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": preprocess_input
        }
    )


@st.cache_data
def load_class_names():
    with open("model/class_names_43.txt", "r") as file:
        return file.read().splitlines()


try:
    model = load_model()
    class_names = load_class_names()

    if len(class_names) != model.output_shape[-1]:
        st.error("Model output count does not match the class names file.")
        st.stop()

except Exception as e:
    st.error(f"Could not load AI model: {e}")
    st.stop()


# ==========================================
# NUTRITION DATABASE
# APPROXIMATE VALUES PER 100 GRAMS
# ==========================================

nutrition = {
    "biryani": {
        "calories": 150, "protein": 5, "carbs": 25, "fat": 4
    },
    "burger": {
        "calories": 295, "protein": 17, "carbs": 24, "fat": 14
    },
    "masala_dosa": {
        "calories": 168, "protein": 4, "carbs": 28, "fat": 4
    },
    "pizza": {
        "calories": 266, "protein": 11, "carbs": 33, "fat": 10
    },
    "samosa": {
        "calories": 262, "protein": 5, "carbs": 31, "fat": 14
    }
}


# ==========================================
# MAIN TABS
# ==========================================

scanner_tab, dashboard_tab = st.tabs([
    "📸 Food Scanner",
    "📊 Nutrition Dashboard"
])


# ==========================================
# FOOD SCANNER
# ==========================================

with scanner_tab:

    st.markdown(
        "<p class='section-heading'>📸 Scan Your Food</p>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:

        st.markdown("#### Upload Food Image")

        uploaded_file = st.file_uploader(
            "Choose a food photo",
            type=["jpg", "jpeg", "png"],
            key="food_upload"
        )

        st.markdown("#### Or Capture Live Photo")

        camera_image = st.camera_input(
            "Take a picture of your food",
            key="food_camera"
        )

        image_source = (
            uploaded_file
            if uploaded_file is not None
            else camera_image
        )

    with right:

        st.markdown("#### 🍽️ How it works")

        st.markdown("""
        <div class="info-card">
            <h3>1. 📷 Upload</h3>
            <p class="small-muted">Choose a clear photo of your food.</p>
            <hr>
            <h3>2. 🤖 Recognize</h3>
            <p class="small-muted">AI analyzes the image and predicts the food.</p>
            <hr>
            <h3>3. 🥗 Explore</h3>
            <p class="small-muted">Check estimated nutrition and save your food record.</p>
        </div>
        """, unsafe_allow_html=True)

    # FOOD RECOGNITION

    if image_source:

        try:

            image = Image.open(image_source).convert("RGB")

            st.divider()

            col_image, col_result = st.columns([1, 1.2])

            with col_image:
                st.markdown("#### 🖼️ Your Food Image")
                st.image(image, use_container_width=True)

            with st.spinner("🤖 FoodAI is analyzing your image..."):

                resized = image.resize((224, 224))
                image_array = np.array(resized)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

            top_indices = np.argsort(predictions)[::-1][:5]

            detected_food = class_names[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)

            with col_result:

                st.markdown("#### 🤖 AI Recognition Result")

                st.markdown(f"""
                <div class="food-result">
                    <p class="small-muted">AI DETECTED FOOD</p>
                    <h2>🍛 {detected_food.replace('_', ' ').title()}</h2>
                    <p><b>Confidence: {confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Top 5 Predictions")

                for index in top_indices:

                    label = class_names[index]
                    score = float(predictions[index])

                    st.write(
                        f"**{label.replace('_', ' ').title()}** — {score * 100:.2f}%"
                    )

                    st.progress(min(max(score, 0.0), 1.0))

            # FOOD CONFIRMATION

            st.divider()

            st.markdown(
                "<p class='section-heading'>🍱 Confirm Your Food</p>",
                unsafe_allow_html=True
            )

            selected_food = st.selectbox(
                "Confirm or change the detected food",
                class_names,
                index=class_names.index(detected_food),
                format_func=lambda food: food.replace("_", " ").title()
            )

            weight = st.number_input(
                "⚖️ Food quantity (grams)",
                min_value=1,
                max_value=2000,
                value=100,
                step=10
            )

            # NUTRITION

            st.markdown(
                "<p class='section-heading'>🥗 Nutrition Information</p>",
                unsafe_allow_html=True
            )

            if selected_food in nutrition:

                values = nutrition[selected_food]

                calories = values["calories"] * weight / 100
                protein = values["protein"] * weight / 100
                carbs = values["carbs"] * weight / 100
                fat = values["fat"] * weight / 100

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("🔥 Calories", f"{calories:.1f} kcal")
                c2.metric("💪 Protein", f"{protein:.1f} g")
                c3.metric("🍞 Carbs", f"{carbs:.1f} g")
                c4.metric("🥑 Fat", f"{fat:.1f} g")

                if st.button(
                    "💾 Save Food to My Dashboard",
                    use_container_width=True
                ):

                    save_food(
                        selected_food,
                        weight,
                        calories,
                        protein,
                        carbs,
                        fat
                    )

                    st.success("✅ Your food record has been saved!")
                    st.balloons()

            else:

                st.info(
                    "Nutrition information is not yet available for this food category. "
                    "We will expand the nutrition database in a future update."
                )

            st.caption(
                "Nutrition values are approximate estimates. "
                "Actual nutrition depends on ingredients and preparation."
            )

        except Exception as e:
            st.error(f"Food recognition error: {e}")

    else:

        st.markdown("""
        <div class="info-card">
            <h2>🍕 Ready to discover your food?</h2>
            <p class="small-muted">
                Upload a food image or capture one with your camera.
                FoodAI will identify the most likely food category.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# NUTRITION DASHBOARD
# ==========================================

with dashboard_tab:

    st.markdown(
        "<p class='section-heading'>📊 Your Daily Nutrition Dashboard</p>",
        unsafe_allow_html=True
    )

    st.caption("Your saved food records for today.")

    history = get_food_history()

    today = [
        record for record in history
        if str(record[7])[:10] == date.today().isoformat()
    ]

    if today:

        total_calories = sum(float(row[3]) for row in today)
        total_protein = sum(float(row[4]) for row in today)
        total_carbs = sum(float(row[5]) for row in today)
        total_fat = sum(float(row[6]) for row in today)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🔥 Calories", f"{total_calories:.1f} kcal")
        c2.metric("💪 Protein", f"{total_protein:.1f} g")
        c3.metric("🍞 Carbs", f"{total_carbs:.1f} g")
        c4.metric("🥑 Fat", f"{total_fat:.1f} g")

        st.divider()

        st.markdown("#### 🍱 Today's Food History")

        food_data = [
            {
                "Food": row[1].replace("_", " ").title(),
                "Quantity (g)": row[2],
                "Calories": row[3],
                "Protein (g)": row[4],
                "Carbs (g)": row[5],
                "Fat (g)": row[6],
                "Date": row[7]
            }
            for row in today
        ]

        st.dataframe(
            pd.DataFrame(food_data),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        chart_left, chart_right = st.columns(2)

        with chart_left:

            st.markdown("#### 📈 Nutrition Breakdown")

            nutrition_data = pd.DataFrame({
                "Nutrient": [
                    "Protein",
                    "Carbohydrates",
                    "Fat"
                ],
                "Amount (g)": [
                    total_protein,
                    total_carbs,
                    total_fat
                ]
            })

            st.bar_chart(
                nutrition_data.set_index("Nutrient")
            )

        with chart_right:

            st.markdown("#### 🔥 Calories by Food")

            calorie_data = pd.DataFrame(food_data)

            st.bar_chart(
                calorie_data.set_index("Food")["Calories"]
            )

    else:

        st.markdown("""
        <div class="info-card">
            <h2>🥗 Your food diary is waiting!</h2>
            <p class="small-muted">
                You have not saved any food records today.
                Scan your first meal to start tracking your nutrition.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    <b>🍽️ FoodAI - Smart Food Recognition System</b><br>
    43 Food Categories | TensorFlow | Streamlit | SQLite<br>
    Made with ❤️ for smarter food tracking
</div>
""", unsafe_allow_html=True)
