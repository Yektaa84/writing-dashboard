#!/usr/bin/env python
# coding: utf-8

# In[34]:


# ================================================================
# 📝 Student Writing Progress Dashboard
# Author: Yekta Ansari
# For: Writing Class Professor
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import textstat
import spacy
import numpy as np
from datetime import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="📝 Student Writing Progress Dashboard",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# --- LOAD SPACY ---
@st.cache_resource
def load_nlp():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        return spacy.load('en_core_web_sm')

nlp = load_nlp()

# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def analyze_text(text):
    """Extract all features from a text"""
    doc = nlp(text)

    tokens = [token for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)

    flesch = textstat.flesch_reading_ease(text)
    gunning = textstat.gunning_fog(text)
    reading_time = textstat.reading_time(text, ms_per_char=14.69)

    lemmas = [token.lemma_.lower() for token in tokens]
    unique_lemmas = len(set(lemmas))
    lexical_diversity = unique_lemmas / len(tokens) if tokens else 0

    nouns = sum(1 for token in tokens if token.pos_ == "NOUN")
    verbs = sum(1 for token in tokens if token.pos_ == "VERB")
    adjectives = sum(1 for token in tokens if token.pos_ == "ADJ")

    return {
        'total_words': len(tokens),
        'unique_words': unique_lemmas,
        'lexical_diversity': lexical_diversity,
        'sentence_count': len(sentences),
        'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
        'nouns': nouns,
        'verbs': verbs,
        'adjectives': adjectives,
        'flesch_score': flesch,
        'gunning_score': gunning,
        'reading_time': reading_time,
        'noun_verb_ratio': nouns / verbs if verbs > 0 else 0
    }

def classify_readability(score):
    if score >= 70:
        return "Easy", "🟢"
    elif score >= 60:
        return "Standard", "🟡"
    else:
        return "Difficult", "🔴"

def analyze_progress(student_data):
    """Analyzes student improvement over time"""
    if len(student_data) < 2:
        return {
            'improvement': 0,
            'status': 'Need at least 2 writings',
            'trend': 'Not enough data',
            'first_score': 0,
            'last_score': 0
        }

    student_data = student_data.sort_values('week')
    first_score = student_data.iloc[0]['flesch_score']
    last_score = student_data.iloc[-1]['flesch_score']
    improvement = last_score - first_score

    if improvement > 5:
        trend = "✅ Improving"
    elif improvement > -5:
        trend = "📊 Stable"
    else:
        trend = "⚠️ Declining"

    return {
        'first_score': first_score,
        'last_score': last_score,
        'improvement': improvement,
        'trend': trend,
        'status': f"{trend} ({improvement:+.1f} points)"
    }

def generate_student_report(student_name, df):
    """Generates a complete report for a student"""
    student_data = df[df['student'] == student_name].sort_values('week')

    if student_data.empty:
        return "No data available for this student"

    progress = analyze_progress(student_data)
    latest = student_data.iloc[-1]

    report = f"""
📝 **STUDENT WRITING REPORT**
**Student:** {student_name}
**Total Writings:** {len(student_data)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **PROGRESS SUMMARY**
- First Writing Score: {progress['first_score']:.1f}
- Latest Writing Score: {progress['last_score']:.1f}
- Change: {progress['improvement']:+.1f} points
- Status: {progress['status']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 **LATEST WRITING METRICS**
- Total Words: {latest['total_words']}
- Sentences: {latest['sentence_count']}
- Avg Sentence Length: {latest['avg_sentence_length']:.1f}
- Lexical Diversity: {latest['lexical_diversity']:.2f}
- Flesch Score: {latest['flesch_score']:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 **WRITING STYLE**
- Nouns: {latest['nouns']}
- Verbs: {latest['verbs']}
- Adjectives: {latest['adjectives']}
- Noun/Verb Ratio: {latest['noun_verb_ratio']:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 **RECOMMENDATIONS**
"""
    if latest['flesch_score'] < 60:
        report += "- 📖 Work on sentence structure and clarity\n"
    if latest['lexical_diversity'] < 0.5:
        report += "- 📚 Try to use more varied vocabulary\n"
    if latest['avg_sentence_length'] > 20:
        report += "- ✂️ Consider breaking up long sentences\n"
    if progress['improvement'] < 0:
        report += "- 📉 Focus on maintaining your writing quality\n"
    if progress['improvement'] > 0:
        report += "- ✅ Keep up the good progress!\n"

    return report

def class_summary(df):
    """Creates a summary dashboard for the whole class"""
    total_students = len(df['student'].unique())
    total_writings = len(df)
    avg_readability = df['flesch_score'].mean()
    avg_lexical = df['lexical_diversity'].mean()
    avg_words = df['total_words'].mean()

    struggling_students = df.groupby('student')['flesch_score'].mean()
    struggling = struggling_students[struggling_students < 60].index.tolist()

    return {
        'total_students': total_students,
        'total_writings': total_writings,
        'avg_readability': avg_readability,
        'avg_lexical': avg_lexical,
        'avg_words': avg_words,
        'struggling_students': struggling,
        'summary': f"""
📊 **CLASS SUMMARY REPORT**
**Total Students:** {total_students}
**Total Writings:** {total_writings}

**Class Averages:**
- Readability: {avg_readability:.1f}
- Lexical Diversity: {avg_lexical:.2f}
- Words per Writing: {avg_words:.0f}

**Students Needing Attention:** {len(struggling)}
"""
    }

# ================================================================
# DATA LOADING
# ================================================================

def create_sample_data():
    """Sample data for testing"""
    students = ['Alice', 'Bob', 'Charlie']
    sample_texts = {
        'Alice': [
            "This is Alice's first week of writing. She is learning how to write better sentences. This is a good start for her.",
            "Alice has been practicing writing every day. Her sentences are becoming longer and more complex. She is improving."
        ],
        'Bob': [
            "Bob is new to writing. He writes short sentences. He is learning.",
            "Bob's writing has improved. He is using more words now. He is more confident."
        ],
        'Charlie': [
            "Charlie is a good writer. He writes about interesting topics. He uses good vocabulary.",
            "Charlie is improving his writing skills. He is using more complex sentences now."
        ]
    }

    data = []
    for student in students:
        for week, text in enumerate(sample_texts[student], start=1):
            features = analyze_text(text)
            data.append({
                'student': student,
                'week': week,
                'text': text,
                **features
            })
    return pd.DataFrame(data)

@st.cache_data
def get_sample_data():
    """Cache only the expensive sample-data generation"""
    return create_sample_data()

def load_data():
    """Load real data if available, otherwise use sample — NOT cached itself"""
    if 'uploaded_df' in st.session_state and st.session_state['uploaded_df'] is not None:
        return st.session_state['uploaded_df']
    return get_sample_data()

# ================================================================
# MAIN APP
# ================================================================

# Load data
df = load_data()

# --- SIDEBAR MENU ---
# --- SIDEBAR MENU ---
with st.sidebar:
    st.title("📝 Navigation")

    # Menu selection
    menu_options = [
        "📊 Dashboard Home",
        "📈 Class Summary", 
        "👤 Student Report",
        "📤 Upload Data",
        "📥 Export Data"
    ]

    selected_menu = st.radio(
        "Choose a section:",
        menu_options,
        index=0
    )

    st.markdown("---")

    # Filters (appear on multiple pages)
    if selected_menu != "📤 Upload Data":
        students = df['student'].unique()
        selected_students = st.multiselect(
            "Select Students",
            options=students,
            default=students
        )

        min_week = int(df['week'].min())
        max_week = int(df['week'].max())

        if min_week < max_week:
            week_range = st.slider(
                "Week Range",
                min_value=min_week,
                max_value=max_week,
                value=(min_week, max_week)
            )
        else:
            week_range = (min_week, max_week)
            st.info(f"📅 Data for week {min_week} only.")

        # Filter data
        filtered_df = df[
            (df['student'].isin(selected_students)) &
            (df['week'] >= week_range[0]) &
            (df['week'] <= week_range[1])
        ]

    st.markdown("---")
    st.caption("👩‍🏫 Created by Yekta Ansari")
    st.caption("📊 Writing Class Dashboard")

# ================================================================
# PAGE: DASHBOARD HOME
# ================================================================
if selected_menu == "📊 Dashboard Home":
    st.title("📝 Student Writing Progress Dashboard")
    st.markdown("Welcome to your writing class dashboard! Use the menu on the left to navigate.")
    st.markdown("---")

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📚 Total Students", len(df['student'].unique()))
    with col2:
        st.metric("📝 Total Writings", len(df))
    with col3:
        st.metric("📖 Avg Readability", f"{df['flesch_score'].mean():.1f}")
    with col4:
        st.metric("📊 Avg Lexical Diversity", f"{df['lexical_diversity'].mean():.2f}")

    # Progress chart
    st.subheader("📈 Class Progress Over Time")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get filtered data (if students selected)
    if 'filtered_df' in locals() and not filtered_df.empty:
        plot_df = filtered_df
    else:
        plot_df = df

    for student in plot_df['student'].unique():
        student_data = plot_df[plot_df['student'] == student]
        ax.plot(student_data['week'], student_data['flesch_score'], 
                marker='o', linewidth=2, label=student)

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Flesch Readability Score', fontsize=12)
    ax.set_title('Readability Progress by Student', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Student table
    st.subheader("📊 Student Overview")
    st.dataframe(
        df.groupby('student').agg({
            'flesch_score': 'mean',
            'lexical_diversity': 'mean',
            'total_words': 'mean',
            'week': 'count'
        }).round(2).rename(columns={'week': 'writings'})
    )

# ================================================================
# PAGE: CLASS SUMMARY
# ================================================================
elif selected_menu == "📈 Class Summary":
    st.title("📊 Class Summary Dashboard")

    # Get class summary
    summary = class_summary(df)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👨‍🎓 Total Students", summary['total_students'])
    with col2:
        st.metric("📝 Total Writings", summary['total_writings'])
    with col3:
        st.metric("⚠️ Students Needing Attention", len(summary['struggling_students']))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📖 Avg Readability", f"{summary['avg_readability']:.1f}")
    with col2:
        st.metric("📊 Avg Lexical Diversity", f"{summary['avg_lexical']:.2f}")
    with col3:
        st.metric("📝 Avg Words", f"{summary['avg_words']:.0f}")

    # Struggling students
    if summary['struggling_students']:
        st.subheader("⚠️ Students Needing Attention")
        st.warning(f"These students have average readability below 60:")
        for student in summary['struggling_students']:
            st.write(f"- {student}")

    # Class distribution charts
    st.subheader("📊 Class Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['flesch_score'].hist(bins=20, color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel('Flesch Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Readability Distribution')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['lexical_diversity'].hist(bins=20, color='lightgreen', edgecolor='black', ax=ax)
        ax.set_xlabel('Lexical Diversity')
        ax.set_ylabel('Frequency')
        ax.set_title('Lexical Diversity Distribution')
        st.pyplot(fig)

# ================================================================
# PAGE: STUDENT REPORT
# ================================================================
elif selected_menu == "👤 Student Report":
    st.title("👤 Student Report")

    # Select student
    all_students = df['student'].unique()
    selected_student = st.selectbox(
        "Choose a student to view detailed report",
        options=all_students
    )

    if selected_student:
        # Get student data
        student_data = df[df['student'] == selected_student].sort_values('week')

        # Show report
        report = generate_student_report(selected_student, df)
        st.markdown(report)

        # Individual progress chart
        st.subheader(f"📈 {selected_student}'s Progress")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(student_data['week'], student_data['flesch_score'], 
                marker='o', linewidth=2, markersize=8, color='blue')
        ax.set_xlabel('Week')
        ax.set_ylabel('Flesch Score')
        ax.set_title(f"{selected_student}'s Readability Progress")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Show all writings
        st.subheader(f"📄 {selected_student}'s Writings")
        for week in student_data['week']:
            with st.expander(f"Week {week}"):
                text = student_data[student_data['week'] == week]['text'].iloc[0]
                st.write(text)

                # Show metrics for this week
                col1, col2, col3 = st.columns(3)
                row = student_data[student_data['week'] == week].iloc[0]
                with col1:
                    st.metric("Words", row['total_words'])
                with col2:
                    st.metric("Flesch Score", f"{row['flesch_score']:.1f}")
                with col3:
                    st.metric("Lexical Diversity", f"{row['lexical_diversity']:.2f}")

# ================================================================
# PAGE: UPLOAD DATA
# ================================================================
elif selected_menu == "📤 Upload Data":
    st.title("📤 Upload Student Data")
    st.markdown("Upload a CSV file with student writings to analyze.")
    st.markdown("---")

    st.info("""
    **CSV Format Required:**

    | student | week | text |
    |---------|------|------|
    | Alice | 1 | "This is Alice's writing..." |
    | Alice | 2 | "Alice's second writing..." |
    | Bob | 1 | "Bob's writing..." |

    **Instructions:**
    1. Save your data as CSV
    2. Upload using the button below
    3. The dashboard will automatically analyze the texts
    4. Use the menu to view results
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: student, week, text"
    )

    if uploaded_file is not None:
        try:
            # Try different encodings
            try:
                new_df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                new_df = pd.read_csv(uploaded_file, encoding='latin-1')

            # Check required columns
            required = ['student', 'week', 'text']
            missing = [col for col in required if col not in new_df.columns]

            if missing:
                st.error(f"Missing columns: {missing}. Please check your CSV format.")
            else:
                # Process the data
                with st.spinner("Analyzing texts..."):
                    features = []
                    for _, row in new_df.iterrows():
                        text_analysis = analyze_text(row['text'])
                        features.append(text_analysis)
                    features_df = pd.DataFrame(features)
                    new_df = pd.concat([new_df, features_df], axis=1)

                    # ✅ Save to session state with the RIGHT variable name
                    st.session_state['df'] = new_df
                    st.session_state['uploaded_df'] = new_df  # ✅ Also save here
                    st.session_state['data_loaded'] = True

                    # Also save to file
                    new_df.to_csv('student_writings.csv', index=False)
                    st.success(f"✅ Data uploaded and analyzed successfully! {len(new_df)} rows loaded.")

                    # Show preview
                    st.subheader("📊 Data Preview")
                    st.dataframe(new_df.head())

                    st.info("Now go to other menu options to view your data!")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # ✅ Show current data if loaded
    if st.session_state.get('data_loaded', False):
        st.subheader("📊 Current Data")
        st.dataframe(st.session_state['uploaded_df'])

        if st.button("🗑️ Clear Data"):
            st.session_state['df'] = None
            st.session_state['uploaded_df'] = None
            st.session_state['data_loaded'] = False
            st.rerun()


# ================================================================
# PAGE: EXPORT DATA
# ================================================================
elif selected_menu == "📥 Export Data":
    st.title("📥 Export Data")
    st.markdown("Download your data in different formats.")

    st.subheader("📊 Full Data Export")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Data (CSV)",
        data=csv,
        file_name="writing_data_export.csv",
        mime="text/csv"
    )

    # Export student-specific data
    st.subheader("👤 Export Student-Specific Data")
    export_student = st.selectbox(
        "Select a student to export their data",
        options=df['student'].unique()
    )

    if export_student:
        student_export = df[df['student'] == export_student]
        csv_student = student_export.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {export_student}'s Data (CSV)",
            data=csv_student,
            file_name=f"{export_student}_data_export.csv",
            mime="text/csv"
        )

    # Summary report
    st.subheader("📄 Summary Report")
    summary = class_summary(df)
    st.markdown(summary['summary'])


# In[ ]:




