This is for TOIS paper: Towards Question-Based High-Recall Information Retrieval: Locating the Last Few Relevant Documents for Technology Assisted Reviews

1. Original SBSTAR: SBSTAR_Main.java 
2. SBSTAR with noise model: SBSTAR_noise.java 
3. SBSTAR_ext model for when-to-stop: SBSTAR_ext.java 

To run this code, you need

(1) define two parameters: the numbers of questions and stop point

(2) the doc-entity results file from entity linking algorithm, e.g., TAGME. Each row is the extracted entities of a document

(3) the missing relevant documents ID for each topic and each stop point

(4) the The initial \alpha of the prior belief P_0 when stopping BMI.
