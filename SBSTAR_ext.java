import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.functions.*;
import weka.classifiers.trees.*;
//import libsvm.svm_print_interface;

public class SBSTAR_ext {

	static ArrayList<String> enList = new ArrayList<String>();
	static ArrayList<String> docList = new ArrayList<String>();
	static ArrayList<String> doc_qrel_relevance = new ArrayList<String>();
	static ArrayList<Integer> questionsIndex = new ArrayList<Integer>();
	static int asked;
	static String questionanswer;
	static int askdyesorno;
	static int countdiscard = 0;
	static ArrayList<Double> stoppingpoint = new ArrayList<Double>();
	static ArrayList<Integer> is_yesorno = new ArrayList<Integer>();
	static ArrayList<Integer> yesornoquestionsnumbers = new ArrayList<Integer>();
	static ArrayList<Integer> Ucandidatesnumbers = new ArrayList<Integer>();
	static ArrayList<Integer> Udifference = new ArrayList<Integer>();
	static ArrayList<Double> pereferencedifference = new ArrayList<Double>();
	static ArrayList<Integer> changetop1 = new ArrayList<Integer>();

	static ArrayList<Integer> questionsnumbers = new ArrayList<Integer>();
	static ArrayList<Integer> is_yes = new ArrayList<Integer>();
	static ArrayList<Integer> is_no = new ArrayList<Integer>();
	static ArrayList<Integer> numbermorethan100 = new ArrayList<Integer>();

	static int maxIteration = 300; // max Iteration parameter

	public static void main(String[] args) throws Exception {
		try {
			String numofquestionsoutput = "/clef/numofquestions.csv"; // output number of questions
			FileWriter numofquestionsoutputwriter = new FileWriter(numofquestionsoutput);
			BufferedWriter numofquestionsoutputbw = new BufferedWriter(numofquestionsoutputwriter);

			File inputFile = new File("whentostop_train_doc.csv"); // the inputed training file. either abstract or document level
			CSVLoader loader = new CSVLoader();
			loader.setFile(inputFile);
			Instances insTrain = loader.getDataSet();
			insTrain.setClassIndex(insTrain.numAttributes() - 1);

			LibSVM classifier = new LibSVM(); // SVM classifier
			// Logistic classifier=new Logistic(); //Logistic classifier
			// Classifier classifier=new RandomForest();// RandomForest
			// classifier
			// MultilayerPerceptron classifier=new MultilayerPerceptron(); //
			// neural network classifier

			classifier.buildClassifier(insTrain);

			String stopDic = "0.35";// stop point
			String topicname = "CD009185";// e.g., topic ID CD009185

			enList = new ArrayList<String>();
			docList = new ArrayList<String>();
			doc_qrel_relevance = new ArrayList<String>();
			questionsIndex = new ArrayList<Integer>();
			asked = 0;
			askdyesorno = 0;

			Map<String, Double> mapAlpha = new HashMap<String, Double>();
			String pathin = "/clef/Entitylinking/doc-entity-results-" + topicname + ".txt";// the doc-entity results file from entity linking algorithm, e.g., TAGME. Each row is the extracted entities of a document
			String doc_qrel_relePath = "/clef/abs_qrels/" + topicname + stopDic + ".txt";// the missing relevant documents ID for each topic and each stop point, either doc level or abstract level
			String countAlpha = "/clef/X_LR/" + stopDic + "/" + topicname + "LRresult.txt";// the initial \alpha of the prior belief P_0 when stopping BMI, either doc level or abstract level
			String entitylist = "/clef/entitylist-" + topicname + ".txt";// the output file of entitylist
			// String entityDocMatrix = "/clef/entityDocMatrix"+ topicname +
			// ".csv";//the output file of entityDocMatrix
			String resultDir = "/clef/finalResults/" + stopDic;
			String resultFile = resultDir + "/" + topicname + "result.txt";// the output file of results
			File ffen = new File(resultDir);
			if (!ffen.exists())
				ffen.mkdirs();

			// using countAlpha
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

			int[] discardfile = new int[count];

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

			// FileWriter writer2 = new FileWriter(new File(entityDocMatrix));
			// BufferedWriter bw2 = new BufferedWriter(writer2);
			// for (int x = 0; x < entityDocMatrixArray.length; x++) {
			// for (int y = 0; y < entityDocMatrixArray[x].length; y++) {
			// bw2.write(Integer.toString(entityDocMatrixArray[x][y]) + ",");
			// }
			// bw2.newLine();
			// }
			// bw2.flush();
			// bw2.close();

			FileReader reader4 = new FileReader(doc_qrel_relePath);
			BufferedReader br4 = new BufferedReader(reader4);
			String str4 = "";
			while ((str4 = br4.readLine()) != null) {
				String docqrelTemp = str4.split(",")[0].substring(0, str4.split(",")[0].indexOf(".txt"));
				doc_qrel_relevance.add(docqrelTemp.trim());
			}
			br4.close();

			int lastcountdiscard = 0;
			int lastmaxindex = 0;
			if (!doc_qrel_relevance.isEmpty()) {
				double if_continue = 0.0;
				for (int w = 0; w <= maxIteration; w++) {
					if (if_continue == 0.0) {

						PrefPi = SequentialBayesianSearch(entityDocMatrixArray, Alpha, discardfile);
						if (w != maxIteration) {
							String whentostopcsv = "/clef/whentostop_test_abs.csv"; // temp file
							FileWriter whentostopwriter = new FileWriter(whentostopcsv);
							BufferedWriter whentostopbw = new BufferedWriter(whentostopwriter);
							whentostopbw.write("stoppingpoint" + "," + "effective_question_numbers" + "," + "is_yesorno"
									+ "," + "GBS_candidates_left" + "," + "U_difference" + "," + "preference_difference"
									+ "," + "change_top1" + "," + "rels" + "\n");
							stoppingpoint.add(Double.parseDouble(stopDic.substring(8)));
							whentostopbw.write(stopDic.substring(8) + ",");
							yesornoquestionsnumbers.add(askdyesorno);
							whentostopbw.write(askdyesorno + ",");
							if (questionanswer.equals("yes")) {
								is_yesorno.add(1);
								whentostopbw.write(1 + ",");
							} else if (questionanswer.equals("no")) {
								is_yesorno.add(1);
								whentostopbw.write(1 + ",");
							} else {
								is_yesorno.add(0);
								whentostopbw.write(0 + ",");
							}
							int countdiscard = 0;
							for (int m = 0; m < discardfile.length; m++)
								countdiscard = countdiscard + discardfile[m];
							Ucandidatesnumbers.add(count - countdiscard);
							whentostopbw.write((count - countdiscard) + ",");
							Udifference.add(countdiscard - lastcountdiscard);
							whentostopbw.write((countdiscard - lastcountdiscard) + ",");

							lastcountdiscard = countdiscard;

							int maxindex = 0;
							double maxprefer = 0.0;
							double[] newPreferencePi = new double[Alpha.length];
							double newcountAlpha = 0.0;
							for (int f = 0; f < Alpha.length; f++) {
								newcountAlpha += Alpha[f];
							}
							for (int q = 0; q < Alpha.length; q++) {
								newPreferencePi[q] = Alpha[q] / newcountAlpha;
								if (newPreferencePi[q] >= maxprefer) {
									maxprefer = newPreferencePi[q];
									maxindex = q;
								}

							}
							double pdiffer = 0.0;
							for (int n = 0; n < Alpha.length; n++) {
								pdiffer = pdiffer + Math.abs(newPreferencePi[n] - PrefPi[n]);
							}
							pereferencedifference.add(pdiffer);
							whentostopbw.write(pdiffer + ",");

							if (maxindex != lastmaxindex) {
								changetop1.add(1);
								whentostopbw.write(1 + ",");
							} else {
								changetop1.add(0);
								whentostopbw.write(0 + ",");
							}

							lastmaxindex = maxindex;

							whentostopbw.write("continue" + "\n");
							whentostopbw.flush();
							whentostopbw.close();

							Instance ins = insTrain.instance(0);
							ins.setValue(0, stoppingpoint.get(stoppingpoint.size() - 1));
							ins.setValue(1, yesornoquestionsnumbers.get(yesornoquestionsnumbers.size() - 1));
							ins.setValue(2, is_yesorno.get(is_yesorno.size() - 1));
							ins.setValue(3, Ucandidatesnumbers.get(Ucandidatesnumbers.size() - 1));
							ins.setValue(4, Udifference.get(Udifference.size() - 1));
							ins.setValue(5, pereferencedifference.get(pereferencedifference.size() - 1));
							ins.setValue(6, changetop1.get(changetop1.size() - 1));

							if_continue = classifier.classifyInstance(ins);
							// System.out.println(logic.classifyInstance(ins));
						} else
							numofquestionsoutputbw.write(stopDic + "," + topicname + "," + w + "\n");
					} else {
						numofquestionsoutputbw.write(stopDic + "," + topicname + "," + w + "\n");
						PrefPi = SequentialBayesianSearch(entityDocMatrixArray, Alpha, discardfile);
						break;
					}
				}
			}
			// System.out.println(topicname+"done");

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

			numofquestionsoutputbw.flush();
			numofquestionsoutputbw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static double[] SequentialBayesianSearch(int[][] entityDocMatrixArray, double[] Alpha, int[] discardfile) {
		double[] PreferencePi = new double[Alpha.length];

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
					if (discardfile[i] != 1) {
						if (entityDocMatrixArray[z][i] == 1)
							argMin = argMin + PreferencePi[i];
						if (entityDocMatrixArray[z][i] == 0)
							argMin = argMin - PreferencePi[i];
					}
				}

				double judge = Math.abs(argMin);// judge if it is minimal

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

		if (countIsZero == countIsZero + countIsOne && countIsZero != 0) {
			askdyesorno++;
			questionanswer = "no";
			// System.out.println(enList.get(minIndex)+"*****NO");
			for (int m = 0; m < entityDocMatrixArray[minIndex].length; m++) {
				if (entityDocMatrixArray[minIndex][m] == 0) {
					Alpha[m] = Alpha[m] + 1;
				} else {
					discardfile[m] = 1;
				}

			}
		}

		else if (countIsOne == countIsZero + countIsOne && countIsOne != 0) {
			askdyesorno++;
			questionanswer = "yes";
			// System.out.println(enList.get(minIndex)+"*********YES");
			for (int m1 = 0; m1 < entityDocMatrixArray[minIndex].length; m1++) {
				if (entityDocMatrixArray[minIndex][m1] == 1) {
					Alpha[m1] = Alpha[m1] + 1;
				} else {
					discardfile[m1] = 1;
				}
			}
		} else {
			questionanswer = "not_sure";
			// System.out.println(enList.get(minIndex)+"*********don't know");
		}
		return PreferencePi;
	}

}
