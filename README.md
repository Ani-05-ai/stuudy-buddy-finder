# stuudy-buddy-finder
<pre>SNU Study Buddy Finder – Project Report
1. Problem Understanding & Assumptions

Problem Statement:
Students at Sister Nivedita University (SNU) often face difficulties finding compatible study partners. This project aims to develop a buddy recommendation system that suggests students with compatible preferences, habits, and interests to enhance collaborative learning.

Dataset Columns:

teamwork_preference – Measures how much a student prefers group work.

introversion_extraversion – Personality spectrum indicating social tendencies.

books_read_past_year – Number of books read in the past year.

club_top1 – Primary club participation of the student.

weekly_hobby_hours – Hours spent on hobbies per week.

Assumptions:

Students with similar teamwork_preference and introversion_extraversion are likely to collaborate effectively.

Similar reading habits indicate compatible study approaches.

Club involvement and hobbies reflect lifestyle and social compatibility.

Missing values in the dataset are removed to ensure data integrity.

2. Data Preprocessing & Feature Engineering

Steps Taken:

Handling Missing Values:

Numeric missing values filled with mean/median.

Categorical missing values imputed with mode.

Encoding Categorical Variables:

club_top1 was converted to numeric format using one-hot encoding.

Feature Scaling:

All numerical features standardized using StandardScaler for uniformity.

Ensured all columns are of consistent datatype for clustering.

Feature Engineering:

No additional features were created.

Applied PCA for dimensionality reduction to 2D and 3D for visualization purposes.

3. Model Selection & Justification

Algorithm: KMeans Clustering

Justification:

The task is unsupervised as no labeled data exists for compatibility.

KMeans efficiently groups students with similar attributes.

Algorithm is scalable and interpretable, suitable for medium datasets like ours.

Number of Clusters:

Determined using the Elbow Method on the inertia plot.

Selected k = 25, with experimentation on k = 20 and k = 9. Minimal changes were observed in the Silhouette Score.

Evaluation:

Silhouette Score: 0.43 → Moderate clustering quality.

Interpretation: Some clusters are well-defined, while overlaps indicate partial compatibility. The system is a supportive tool rather than an exact solution.

4. Visualization & Interpretation

PCA Scatter Plots (2D & 3D):

PCA used to reduce feature space for visualization.

Each cluster is color-coded for clarity.

Insights:

Dense clusters suggest strong compatibility among some student groups.

Overlapping clusters indicate areas where students may fit multiple groups, reflecting diverse preferences.

5. Insights & Recommendations

Students within the same cluster can be recommended as study buddies.

Integrate club membership to further enhance compatibility (students from the same club often collaborate better).

Match students with similar hobby hours to improve engagement and productivity.

The moderate Silhouette Score suggests that the system supports human judgment rather than fully automating pairings.

6. Conclusion

The SNU Study Buddy Finder provides a data-driven approach to pairing students based on preferences, personality traits, and interests. Using KMeans clustering and PCA visualizations, compatible groups were identified, though overlaps reflect the natural diversity of student habits. This system serves as a recommendation aid, helping students find suitable study partners while allowing flexibility for human discretion. Future improvements could include additional behavioral or academic features to increase the Silhouette Score and further refine compatibility assessments.</pre>
