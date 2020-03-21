import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class SBSTAR_noise {

	static ArrayList<String> enList = new ArrayList<String>();
	static ArrayList<String> docList = new ArrayList<String>();
	static ArrayList<String> doc_qrel_relevance = new ArrayList<String>();
	static ArrayList<Integer> questionsIndex = new ArrayList<Integer>();
	static int asked;
	static double noise;

	public static void main(String[] args) throws Exception {
		try {
			int iteration1 = 10;// the numbers of questions
			String stopDic = "0.65";//stop point

				enList = new ArrayList<String>();
				docList = new ArrayList<String>();
				doc_qrel_relevance = new ArrayList<String>();
				questionsIndex = new ArrayList<Integer>();
				asked = 0;

				String topicname = "CD007394";//e.g., topic ID CD007394
				noise=0.5;//noise rate
				Map<String, Double> mapAlpha = new HashMap<String, Double>();
				String pathin = "/clef/Entitylinking/doc-entity-results-"+ topicname + ".txt";// the doc-entity results file from entity linking algorithm, e.g., TAGME. Each row is the extracted entities of a document
				String doc_qrel_relePath = "/clef/abs_qrels/" + topicname + stopDic + ".txt";//the missing relevant documents ID for each topic and each stop point, either doc level or abstract level
				String countAlpha = "/clef/X_LR/" + stopDic + "/"+ topicname + "LRresult.txt";//the initial \alpha of the prior belief P_0 when stopping BMI, either doc level or abstract level
				String entitylist = "/clef/entitylist-"+ topicname + ".txt";//the output file of entitylist
				String entityDocMatrix = "/clef/entityDocMatrix"+ topicname + ".csv";//the output file of entityDocMatrix
				String resultDir = "/clef/finalResults/" + stopDic + "/question" + iteration1;
				String resultFile =resultDir+"/"+ topicname + "result.txt";//the output file of results
				File ffen=new File(resultDir);
				if (!ffen.exists())
					ffen.mkdirs();
				
				FileReader reader5 = new FileReader(countAlpha);
				BufferedReader br5 = new BufferedReader(reader5);
				String str5 = "";
				ArrayList<String> docL = new ArrayList<String>();
				while ((str5 = br5.readLine()) != null) {
					String[] docqrelTe = str5.split(",");
					String ssfilename = docqrelTe[0].substring(docqrelTe[0].indexOf("('") + 2,
							docqrelTe[0].lastIndexOf(".txt"));
					String Proba = docqrelTe[1].substring(0, docqrelTe[1].lastIndexOf(")")).trim();
					mapAlpha.put(ssfilename + "txt", Double.parseDouble(Proba));
					docL.add(ssfilename + "txt");
				}

				FileWriter writer = new FileWriter(entitylist);
				BufferedWriter bw = new BufferedWriter(writer);
				FileReader reader1 = new FileReader(pathin);
				BufferedReader br1 = new BufferedReader(reader1);
				String str1 = "";
				int count = 0;
				br1.readLine();// remove the first empty line
				while ((str1 = br1.readLine()) != null) {
					if (docL.contains(str1.split(",")[0].trim())) {
						count++;
						String entities[] = str1.split(",");
						docList.add(entities[0].trim());
						for (int i = 0; i < entities.length; i++) {
							if (!entities[i].trim().equals("") && i > 0) {
								if (!enList.contains(entities[i].trim())) {
									enList.add(entities[i].trim());
									bw.write(entities[i].trim() + "\n");
								}
							}
						}
					}
				}
				bw.flush();
				bw.close();
				br1.close();

				double[] Alpha = new double[count];
				double[] PrefPi = new double[Alpha.length];

				for (int d = 0; d < docList.size(); d++) {
					if (mapAlpha.get(docList.get(d)) != null)
						Alpha[d] = mapAlpha.get(docList.get(d));
					else
						Alpha[d] = 0;
				}
				br5.close();

				int[][] entityDocMatrixArray = new int[enList.size()][count];
				FileReader reader2 = new FileReader(pathin);
				BufferedReader br2 = new BufferedReader(reader2);
				String str2 = "";
				int column1 = 0;
				br2.readLine();
				while ((str2 = br2.readLine()) != null) {
					if (docL.contains(str2.split(",")[0].trim())) {
						String entities2[] = str2.split(",");
						for (int i = 0; i < entities2.length; i++) {
							if (!entities2[i].trim().equals("") && i > 0) {
								entityDocMatrixArray[enList.indexOf(entities2[i].trim())][column1] = 1;
							}
						}
						column1++;
					}
				}
				br2.close();

				FileWriter writer2 = new FileWriter(new File(entityDocMatrix));
				BufferedWriter bw2 = new BufferedWriter(writer2);
				for (int x = 0; x < entityDocMatrixArray.length; x++) {
					for (int y = 0; y < entityDocMatrixArray[x].length; y++) {
						bw2.write(Integer.toString(entityDocMatrixArray[x][y]) + ",");
					}
					bw2.newLine();
				}
				bw2.flush();
				bw2.close();

				FileReader reader4 = new FileReader(doc_qrel_relePath);
				BufferedReader br4 = new BufferedReader(reader4);
				String str4 = "";
				while ((str4 = br4.readLine()) != null) {
					String docqrelTemp = str4.split(",")[0].substring(0, str4.split(",")[0].indexOf(".txt"));
					doc_qrel_relevance.add(docqrelTemp.trim());
				}
				br4.close();

				for (int w = 0; w <= iteration1; w++) { 
					PrefPi = SequentialBayesianSearch(entityDocMatrixArray, Alpha);
				}

				// sorting
				Map<String, Double> map = new HashMap<String, Double>();
				for (int mm = 0; mm < PrefPi.length; mm++) {
					map.put(docList.get(mm), PrefPi[mm]);
				}
				List<Map.Entry<String, Double>> mappingList = new ArrayList<Map.Entry<String, Double>>(map.entrySet());
				Collections.sort(mappingList, new Comparator<Map.Entry<String, Double>>() {
					public int compare(Map.Entry<String, Double> mapping1, Map.Entry<String, Double> mapping2) {
						return (mapping2.getValue()).compareTo(mapping1.getValue());
					}
				});
				// System.out.println("Results:");
				FileWriter writerresult = new FileWriter(resultFile);
				BufferedWriter bwresult = new BufferedWriter(writerresult);
				for (Map.Entry<String, Double> mapping : mappingList) {
					bwresult.write(mapping.getValue().toString() + "=" + mapping.getKey().toString() + "\n");
					// System.out.println(mapping.getValue().toString()+"="+mapping.getKey().toString());
				}
				bwresult.flush();
				bwresult.close();
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	//noise setting 1 and noise setting 2
	public static double[] SequentialBayesianSearch(int[][] entityDocMatrixArray, double[] Alpha) {

		double[] PreferencePi = new double[Alpha.length];
		double randomnoise=Math.random();

		double countAlpha = 0.0;
		for (int f = 0; f < Alpha.length; f++) {
			countAlpha += Alpha[f];
		}

		for (int q = 0; q < Alpha.length; q++) {
			PreferencePi[q] = Alpha[q] / countAlpha;
		}
		double min = 1000000000.0;
		int minIndex = 0;//
		int count2 = 0;
		for (int z = 0; z < entityDocMatrixArray.length; z++) {
			double argMin = 0.0;//
			if (!questionsIndex.contains(z)) {// exclude the already asked questions
				for (int i = 0; i < entityDocMatrixArray[z].length; i++) {
					if (entityDocMatrixArray[z][i] == 1)
						argMin = argMin + PreferencePi[i];
					if (entityDocMatrixArray[z][i] == 0)
						argMin = argMin - PreferencePi[i];
				}

				double judge = Math.abs(argMin);// noise setting 1: judge if it is minimal. 
				//double judge = 2.0 * beta * (1.0 / (2.0 * (1.0 + avgtf.get(count2)))) + Math.abs(argMin); //noise setting 2

				if (judge < min) {
					min = judge;
					minIndex = count2;//
				}
			}
			count2++;

		}
		questionsIndex.add(minIndex);
		asked = minIndex;

		int countIsZero = 0;
		int countIsOne = 0;
		int countNull = 0;

		for (int k = 0; k < doc_qrel_relevance.size(); k++) {
			int indexM = docList.indexOf(doc_qrel_relevance.get(k) + "txt");
			if (indexM != -1) {// if -1, meaning the Abstract is null
				if (entityDocMatrixArray[minIndex][indexM] == 0)
					countIsZero++;
				if (entityDocMatrixArray[minIndex][indexM] == 1)
					countIsOne++;
			} else {
				countNull++;
			}
		}

		//noise = 1.0 / (2.0 * (1.0 + avgtf.get(minIndex))); //noise setting 2

		if (countIsZero == countIsZero + countIsOne && countIsZero != 0) {
			// System.out.println(enList.get(minIndex)+"*****NO");
			for (int m = 0; m < entityDocMatrixArray[minIndex].length; m++) {
					if(0<= randomnoise && randomnoise < 0.5*noise){//noise setting
						if (entityDocMatrixArray[minIndex][m] == 1) {//noise to "yes"
							Alpha[m] = Alpha[m] + 1;
						}
					}
					else if(randomnoise >= 0.5*noise && randomnoise < noise){
						//noise to "not sure"
					}
					else{//correct answer					
						if(entityDocMatrixArray[minIndex][m]==0){
							Alpha[m]=Alpha[m]+1;
						}
					}
			}
		}

		else if (countIsOne == countIsZero + countIsOne && countIsOne != 0) {
			// System.out.println(enList.get(minIndex)+"*********YES");
			for (int m1 = 0; m1 < entityDocMatrixArray[minIndex].length; m1++) {
					if(0<= randomnoise && randomnoise < 0.5*noise){//noise setting
						if (entityDocMatrixArray[minIndex][m1] == 0) {//noise to "no"
							Alpha[m1]=Alpha[m1]+1; 
						}
					}
					else if(randomnoise >= 0.5*noise && randomnoise < noise){
						//noise to "not sure"
					}
					else{//correct answer
						if(entityDocMatrixArray[minIndex][m1]==1){
							Alpha[m1]=Alpha[m1]+1; 
						}
					}

			}
		} else {
			// System.out.println(enList.get(minIndex)+"*********don't know");
			for(int m2=0; m2<entityDocMatrixArray[minIndex].length;m2++){
					if(0<= randomnoise && randomnoise < 0.5*noise){//noise setting
						if (entityDocMatrixArray[minIndex][m2] == 0) { //noise to "no"
							Alpha[m2]=Alpha[m2]+1; 
						}
					}
					else if(randomnoise >= 0.5*noise && randomnoise < noise){//noise to "yes"
						if (entityDocMatrixArray[minIndex][m2] == 1) {
							Alpha[m2]=Alpha[m2]+1; 
						}
					}
					else{//correct answer

					}
					
				}
		}
		return PreferencePi;
	}

//	//noise setting 3
//	public static double[] SequentialBayesianSearch(int[][] entityDocMatrixArray, double[] Alpha) {
//
//		double[] PreferencePi = new double[Alpha.length];
//		double randomnoise=Math.random();
//
//		double countAlpha = 0.0;
//		for (int f = 0; f < Alpha.length; f++) {
//			countAlpha += Alpha[f];
//		}
//
//		for (int q = 0; q < Alpha.length; q++) {
//			PreferencePi[q] = Alpha[q] / countAlpha;
//		}
//		double min = 1000000000.0;
//		int minIndex = 0;//
//		int count2 = 0;
//		for (int z = 0; z < entityDocMatrixArray.length; z++) {
//			double argMin = 0.0;//
//			if (!questionsIndex.contains(z)) {// exclude the already asked questions
//				for (int i = 0; i < entityDocMatrixArray[z].length; i++) {
//					if (entityDocMatrixArray[z][i] == 1)
//						argMin = argMin + PreferencePi[i];
//					if (entityDocMatrixArray[z][i] == 0)
//						argMin = argMin - PreferencePi[i];
//				}
//
//				double judge = Math.abs(argMin);// judge if it is minimal
//
//				if (judge < min) {
//					min = judge;
//					minIndex = count2;//
//				}
//			}
//			count2++;
//
//		}
//		questionsIndex.add(minIndex);
//		asked = minIndex;
//
//		int countIsZero = 0;
//		int countIsOne = 0;
//		int countNull = 0;
//
//		for (int k = 0; k < doc_qrel_relevance.size(); k++) {
//			int indexM = docList.indexOf(doc_qrel_relevance.get(k) + "txt");
//			if (indexM != -1) {// if -1, meaning the Abstract is null
//				if (entityDocMatrixArray[minIndex][indexM] == 0)
//					countIsZero++;
//				if (entityDocMatrixArray[minIndex][indexM] == 1)
//					countIsOne++;
//			} else {
//				countNull++;
//			}
//		}
//
//		if (countIsZero == countIsZero + countIsOne && countIsZero != 0) {
//			// System.out.println(enList.get(minIndex)+"*****NO");
//			for (int m = 0; m < entityDocMatrixArray[minIndex].length; m++) {
//				if (entityDocMatrixArray[minIndex][m] == 0) {
//					Alpha[m] = Alpha[m] + 1;
//				}
//			}
//		}
//
//		else if (countIsOne == countIsZero + countIsOne && countIsOne != 0) {
//			// System.out.println(enList.get(minIndex)+"*********YES");
//			for (int m1 = 0; m1 < entityDocMatrixArray[minIndex].length; m1++) {
//				if (entityDocMatrixArray[minIndex][m1] == 1) {
//					Alpha[m1] = Alpha[m1] + 1; 
//				}
//
//			}
//		} else {
//			// System.out.println(enList.get(minIndex)+"*********don't know");
//			noise=(double)(countIsOne)/(double)(countIsZero+countIsOne);
//			double sn=(double)(countIsZero)/(double)(countIsZero+countIsOne);
//			if(sn<noise)
//				noise=sn;
//			for(int m2=0; m2<entityDocMatrixArray[minIndex].length;m2++){
//					if(0<= randomnoise && randomnoise < 0.5*noise){//noise setting
//						if (entityDocMatrixArray[minIndex][m2] == 0) { //noise to "no"
//							Alpha[m2]=Alpha[m2]+1; 
//						}
//					}
//					else if(randomnoise >= 0.5*noise && randomnoise < noise){//noise to "yes"
//						if (entityDocMatrixArray[minIndex][m2] == 1) {
//							Alpha[m2]=Alpha[m2]+1; 
//						}
//					}
//					else{//correct answer
//
//					}
//					
//				}
//		}
//		return PreferencePi;
//	}

}
