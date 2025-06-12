import streamlit as st
import joblib
import numpy as np
from lightfm import LightFM
import pandas as pd
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sparse_vstack
import scipy.sparse as sp
import gspread
from gspread.exceptions import SpreadsheetNotFound, GSpreadException
from google.oauth2.service_account import Credentials
from datetime import datetime

# =====================
# CUSTOM STYLES
# =====================
st. set_page_config(layout="wide")

# Add Google Fonts
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Merriweather:wght@400;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# =====================
# APP FUNCTIONALITY
# =====================

# Load assets
@st.cache_resource
def load_assets():
    # Load model with custom handler
    model = joblib.load('lightfm_model.pkl')    
    # Load other data
    data = joblib.load('supporting_data.pkl')
    return model, data

model, data = load_assets()

# Access the features
user_features_test = data['user_features_test']
books_df = data['books_df']
users_df = data['users_df']
user_id_mapping = data['user_id_mapping']
item_id_mapping = data['item_id_mapping']
cold_user_ids = data['cold_user_ids']
test_ratings = data['test_ratings']
num_users = len(user_id_mapping)




@st.dialog("Book Details", width="large")
def show_book_details(isbn):
    book = books_df[books_df['ISBN'] == isbn].iloc[0]
    st.image(book['Image-URL-L'] if pd.notna(book['Image-URL-L']) else "https://placehold.co/150x200?text=Cover+Not+Available", use_container_width=True)
    st.subheader(book['Cleaned_Title'])
    st.markdown(f"**Author:** {book['Book-Author']}")
    
    genres = book['genres']
    if isinstance(genres, list):
        genres = ", ".join(genres)
    st.markdown(f"**Genres:** {genres}")

    # Add more detailed info if needed
    st.markdown("---")
    st.markdown("More book details can go here...")


# Define dialog function for book details
# !!!!!!!!!!!find out if we close the button then set the state to null
@st.dialog("Book Description", width="large")
def show_book_details_dialog(isbn):
    book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if pd.notna(book_info.get('Image-URL-L')):
            st.image(book_info['Image-URL-L'], use_container_width=True)
        else:
            st.image("https://placehold.co/300x400?text=Cover+Not+Available", 
                     use_container_width=True)
    
    with col2:
        # Custom styled title (larger font + teal color)
        st.markdown(
            f"<h1 style='font-size: 30px; color: #52c3be;'>{book_info['Cleaned_Title']}</h1>", 
            unsafe_allow_html=True
        )
        
        # Custom styled author (light gray color)
        st.markdown(
            f"<div style='color: #ebefe7; font-size: 22px; margin-bottom: 12px;'>by {book_info['Book-Author']} (Author)</div>", 
            unsafe_allow_html=True
        )
        
        # Genres
        genres = book_info['genres']
        if isinstance(genres, list):
            genre_pills = "".join(
                [f'<span class="genre-pill">{g}</span>' 
                for g in genres[:3]]
            )
            st.markdown(f'<div class="genre-container">{genre_pills}</div>', 
                        unsafe_allow_html=True)
        
        # Series
        if book_info['Series'] and book_info['Series'] != "Standalone":
            st.markdown(
                f"""
                <div style='font-style: italic;'>
                    Part of the {book_info['Series']} series
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("  \n")
        
        # Description
        st.markdown("### Description")
        if book_info.get('description') and not pd.isna(book_info['description']):
            st.markdown(book_info['description'])
        else:
            st.markdown("*No description available.*")

        st.markdown("---")
        st.markdown(f"Published in {int(book_info['Year-Of-Publication'])} by {book_info['Publisher']}")



# =====================
# APP LAYOUT
# =====================

st.image("Book Recommender System.png", use_container_width=True)

st.markdown("  \n")
st.markdown("  \n")
st.markdown("  \n")

st.markdown("""
    <style>
        .hero {
            background-color: #fac826;
            padding: 2rem;
            text-align: center;
            width: 100%;
            margin: 20px 0;
            border-radius: 20px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            color: #3d3a3a !important;
            font-family: 'Playfair Display', serif;
            margin: 0;
        }
        .custom-divider {
            width: 100px;
            height: 4px;
            background-color: #3d3a3a;
            margin: 20px auto;
            border-radius: 2px;
        }
        .hero-description {
            max-width: 700px;
            margin: 0 auto;
            font-size: 1.1rem;
            color: #3d3a3a;
            font-family: 'Arial', sans-serif;
        }
    </style>

    <div class="hero">
        <div class="custom-divider"></div>
        <p class="hero-description">
            Our collection comes from the BookCrossing Community, featuring thousands of books 
            from the golden era of late 20th century literature. While we can't offer the latest bestsellers, 
            we hope to help you discover forgotten masterpieces and hidden gems.
        </p>
    </div>
""", unsafe_allow_html=True)

st.image("dots.png", use_container_width=True) 



# =====================
# PREFERENCES SECTION 
# =====================
with st.container():
    

    left_spacer, content, right_spacer = st.columns([0.05, 0.30, 0.05])
    with content:

        st.image("banner_choose_pref.png", use_container_width=True) 

        st.markdown("  \n")
        st.markdown("  \n")
        st.markdown("  \n")

        col1, col2, col3 = st.columns([1, 2, 1])

            # with col1:
            #     top_genres = ["Fantasy", "Science Fiction", "Romance", "Mystery & Crime", 
            #                 "Nonfiction (General)"]
            #     quick_genres = st.pills(
            #         "Quick Pick: Select favorite genres",
            #         options=top_genres,
            #         selection_mode="multi"
            #     )

            #     custom_genres = st.multiselect(
            #         "Or search for other genres",
            #         options=sorted(data['genre_feature_mapping'].keys())
            #     )

            #     selected_genres = list(set(quick_genres + custom_genres))
            #     if len(selected_genres) > 3:
            #         st.warning("Please select no more than 3 genres")

        with col2:
            with st.form("user_preferences"):
                top_authors = ["Agatha Christie", "John Grisham", "J.K. Rowling", "Stephen King", 
                            "Nora Roberts", "Michael Crichton"]

                quick_authors = st.pills(
                    "Quick Pick: Select 1 to 3 favorite authors",
                    options=top_authors,
                    selection_mode="multi"
                )

                custom_authors = st.multiselect(
                    "Or search for other authors",
                    options=sorted(data['author_feature_mapping'].keys())
                )

                selected_authors = list(set(quick_authors + custom_authors))
                # if len(selected_authors) > 3 or len(selected_authors) == 0:
                #     st.warning("Please select 1 to 3 authors")

                submitted = st.form_submit_button("💾 Save Preferences", use_container_width=True)




# =====================
# RECOMMENDATION CONTROLS
# ===================== 
left_spacer, content, right_spacer = st.columns([0.25, 0.5, 0.25]) 
with content:
    st.markdown("  \n")
    st.markdown("  \n")

    st.image("banner_slider.png", use_container_width=True) 

    st.markdown("  \n")
    st.markdown("  \n")

    num_recommendations = st.slider(
        "**Number of recommendations**", 
        5, 20, 10,
        help="Select how many book suggestions you'd like to see"
    )



    # Generate recommendations button
    if st.button("𝐆𝐞𝐧𝐞𝐫𝐚𝐭𝐞 𝐑𝐞𝐜𝐨𝐦𝐦𝐞𝐧𝐝𝐚𝐭𝐢𝐨𝐧𝐬 💕", type="primary", use_container_width=True):
        # Get dataset mappings
        user_feature_map = data['user_feature_map']
        user_feature_vec = np.zeros(len(user_feature_map))
        
        # Set features
        # for genre in selected_genres:
        #     feature_name = f"genre_{genre.strip()}"
        #     if feature_name in user_feature_map:
        #         user_feature_vec[user_feature_map[feature_name]] = 1.0

        for author in selected_authors:
            feature_name = f"author_{author.strip()}"
            if feature_name in user_feature_map:
                user_feature_vec[user_feature_map[feature_name]] = 1.0

        # Convert to sparse format
        custom_user_features = sp.csr_matrix(user_feature_vec.reshape(1, -1))

        # Generate predictions
        scores = model.predict(
            user_ids=0,
            item_ids=np.arange(len(item_id_mapping)),
            user_features=custom_user_features,
            num_threads=4
        )

        # Get recommendations
        top_indices = np.argsort(-scores)[:num_recommendations]
        isbn_list = list(item_id_mapping.keys())
        recommended_isbns = [isbn_list[idx] for idx in top_indices]

        # Store recommendations in session state
        st.session_state.recommended_isbns = recommended_isbns
        st.session_state.show_recommendations = True



# Check if we have recommendations to show
if "show_recommendations" in st.session_state and st.session_state.show_recommendations:
    
    st.image("banner_rec.png", use_container_width=True) 
 
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown("  \n")

# maybe add like stars/ratings, maybe add heart/love as favorite button, border container?
    # Create grid display
    n_cols = 4
    rows = [st.session_state.recommended_isbns[i:i + n_cols] 
            for i in range(0, len(st.session_state.recommended_isbns), n_cols)]


    for row in rows:
        cols = st.columns(n_cols)
        for col_index, isbn in enumerate(row):
            book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
            
            with cols[col_index]:
                # Create grid card container
                # Inject custom CSS styles at the top of your app
                st.markdown("""
                <style>
                    .book-grid-title {
                        text-align: center;
                        font-size: 1.4rem !important;
                        font-weight: 700 !important;
                        line-height: 1.3;
                        margin: 8px 0 4px 0 !important;
                        color: #52c3be;
                    }
                    .book-grid-author {
                        text-align: center;
                        font-size: 1rem !important;
                        font-style: italic;
                        margin: 0 0 8px 0 !important;
                        color: #ebefe7;
                    }
                    .genre-pill {
                        display: inline-block;
                        background-color: transparent !important; /* Transparent background */
                        color: #ebefe7 !important;              /* Text color */
                        border: 1px solid #275445 !important;   /* Border with same color */
                        padding: 3px 10px !important;
                        border-radius: 16px !important;
                        font-size: 0.8rem !important;
                        margin: 2px 4px 2px 0 !important;
                        white-space: nowrap !important;
                        transition: all 0.2s ease !important;   /* Smooth transitions */
                    }

                    .genre-pill:hover {
                        background-color: rgba(218, 225, 174, 0.1) !important; /* Subtle hover effect */
                        transform: translateY(-1px);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    }

                    .genre-container {
                        text-align: center;
                        margin: 8px 0 12px 0 !important;
                        line-height: 1.5 !important;
                    }
                    
                </style>
                """, unsafe_allow_html=True)

                with st.container():                    
                    # left_pad, center_col, right_pad = st.columns([1, 2, 1])
                    
                    # with center_col:
                    # Book cover (unchanged)
                    if pd.notna(book_info.get('Image-URL-L')):
                        st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <img src="{book_info['Image-URL-L']}" width="200">
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            """
                            <div style='text-align: center;'>
                                <img src="https://placehold.co/150x200?text=Cover+Not+Available" width="200">
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Title with improved truncation
                    title = book_info["Cleaned_Title"]
                    if len(title) > 50:
                        title = title[:47] + "..."
                    st.markdown(f'<div class="book-grid-title">{title}</div>', unsafe_allow_html=True)
                    
                    # Author
                    author = book_info["Book-Author"]
                    if len(author) > 30:
                        author = author[:27] + "..."
                    st.markdown(f'<div class="book-grid-author">by {author}</div>', unsafe_allow_html=True)
                    
                    # Genres as pills
                    genres = book_info['genres']
                    if isinstance(genres, list):
                        genre_pills = "".join(
                            [f'<span class="genre-pill">{g}</span>' 
                            for g in genres[:3]]
                        )
                        st.markdown(f'<div class="genre-container">{genre_pills}</div>', 
                                    unsafe_allow_html=True)
                    
                    # Details button
                    btn_left, btn_center, btn_right = st.columns([1, 2, 1])
                    with btn_center:
                        if st.button("View Details", key=f"detail_{isbn}", use_container_width=True):
                            st.session_state.selected_book = isbn

                    st.markdown("  \n")
                    st.markdown("  \n")
                    st.markdown("  \n")



# Show book details page if a book is selected
if "selected_book" in st.session_state and st.session_state.selected_book:
    show_book_details_dialog(st.session_state.selected_book)




# =====================
# FEEDBACK SECTION
# =====================
st.markdown("  \n")
st.markdown("  \n")
st.markdown("  \n")
st.markdown("  \n")
st.markdown("  \n")


left_spacer, content, right_spacer = st.columns([0.25, 0.6, 0.25]) 
with content:
    st.image("banner_feedback2.png", use_container_width=True)
    st.markdown("  \n")
    st.markdown("  \n")

    with st.container():
        
        with st.form("recommendation_feedback"):
            # Email collection
            email = st.text_input("**Your email**", 
                                placeholder="name@example.com",
                                help="This serves as proof in my thesis that you're not a bot. It will not be used for any other purpose.",)
            
            # Rating scale
            st.markdown("**How are the recommendations? (Is it relevant to you?)**")
            rating = st.radio(
                "",
                [
                    "Excellent! Perfect matches!",
                    "Good! Mostly relevant",
                    "Fair. Got some good suggestions",
                    "Bad. Not what I wanted",
                    "Horrible. Completely off"
                ],
                label_visibility="collapsed"
            )
            
            # Detailed feedback
            feedback_text = st.text_area("**What could we improve?** (optional)", 
                                    placeholder="Your suggestions...",
                                    height=120)
            
            # Form submission
            submitted = st.form_submit_button("📤 Submit Feedback", use_container_width=True)
            
            if submitted:
                rating_map = {
                    "Excellent! Perfect matches!": 5,
                    "Good! Mostly relevant": 4,
                    "Fair. Got some good suggestions": 3,
                    "Bad. Not what I wanted": 2,
                    "Horrible. Completely off": 1,
                }
                
                numerical_rating = rating_map[rating]
                
                # Initialize Google Sheets connection
                def init_gsheets():
                    try:
                        scope = [
                            "https://www.googleapis.com/auth/spreadsheets",
                            "https://www.googleapis.com/auth/drive"
                        ]
                        creds = Credentials.from_service_account_info(
                            st.secrets["gcp_service_account"],
                            scopes=scope
                        )
                        client = gspread.authorize(creds)
                        sheet_id = "115Ou7SNIoQdBde-jc7uQ7w2jDl9N8wDQfupbAKwQZys"
                        sheet = client.open_by_key(sheet_id)
                        return sheet.sheet1
                    except Exception as e:
                        st.error(f"Error connecting to feedback system: {str(e)}")
                        return None
                
                # Save feedback
                def save_feedback(email, rating, feedback_text):
                    try:
                        sheet = init_gsheets()
                        if not sheet:
                            return False
                        
                        sheet.append_row([
                            datetime.now().isoformat(),
                            email or "anonymous",
                            rating,
                            feedback_text,
                            "v2.0"
                        ], value_input_option="USER_ENTERED")
                        
                        return True
                    except Exception as e:
                        st.error(f"Error saving feedback: {str(e)}")
                        return False
                
                if save_feedback(
                    email=email if email else "anonymous",
                    rating=numerical_rating,
                    feedback_text=feedback_text
                ):
                    st.success("🎉 Thank you for your feedback! We appreciate your input.")
                    st.balloons()