package com.UpGrad.Data_Pre_Processing;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class AutoData {

	private static final Dataset<?> SongId = null;

	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		SparkSession sparkSession = SparkSession.builder().appName("SparkML").master("local[*]").getOrCreate();

		Dataset<Row> user_click_stream_activity = sparkSession.read().option("inferschema", true)
				.csv("data/sample100mb.csv").toDF("UserId", "TimeStamp", "SongId", "Date");
		// Reading Data from a CSV file //Inferring Schema and Setting Header as True

		user_click_stream_activity = user_click_stream_activity.drop("TimeStamp", "Date");
		user_click_stream_activity = user_click_stream_activity.na().drop();
		user_click_stream_activity.show();

		// Get the song frequency per user by groupby operation Dataset<Row>
		Dataset<Row> user_click_stream_activity_count = user_click_stream_activity.groupBy("UserId", "SongId").count()
				.toDF("UserId", "SongId", "Frequency"); //
		user_click_stream_activity_count.show();
		user_click_stream_activity_count.orderBy(org.apache.spark.sql.functions.col("Frequency").desc()).show();

		// ***********************************String //
		// Indexer*******************************************************

		// Converting a categorical attribute to a numerical attribute

		StringIndexer indexer = new StringIndexer().setInputCol("UserId").setOutputCol("UserId_Index");

		StringIndexerModel indModel = indexer.fit(user_click_stream_activity_count);

		Dataset<Row> indexedUserKnow = indModel.transform(user_click_stream_activity_count);

		indexedUserKnow.groupBy(col("UserId"), col("UserId_Index")).count().show();
		// indexedUserKnow.where("UserId_Index = 133783").show();

		StringIndexer indexer1 = new StringIndexer().setInputCol("SongId").setOutputCol("SongId_Index");

		StringIndexerModel indModel1 = indexer1.fit(user_click_stream_activity_count);

		Dataset<Row> indexedSongId = indModel1.transform(user_click_stream_activity_count);

		indexedSongId.groupBy(col("SongId"), col("SongId_Index")).count().show();

		indexedSongId.printSchema();
		// indexedUserKnow.where("UserId_Index = 133783").show();

		Dataset<Row> userIndexed = indexer.fit(user_click_stream_activity_count)
				.transform(user_click_stream_activity_count);

		indexer.setInputCol("SongId").setOutputCol("SongId_Index");

		Dataset<Row> songIndexed = indexer.fit(userIndexed).transform(userIndexed);

		Dataset<Row> modelIndexed = songIndexed
				.withColumn("UserId_Index", col("UserId_Index").cast(DataTypes.IntegerType))
				.withColumn("SongId_Index", col("SongId_Index").cast(DataTypes.IntegerType));

		ALS als = new ALS().setRank(10).setMaxIter(5).setRegParam(0.01).setUserCol("UserId_Index")
				.setItemCol("SongId_Index").setRatingCol("Frequency");
		ALSModel model = als.fit(modelIndexed);

		// Get the userFactors from ALS model to use it in kmeans
		Dataset<Row> userALSFeatures = model.userFactors();

		// <UserId,UserIndex>
		Dataset<Row> userIdTable = modelIndexed.drop("SongId_Index", "SongId", "Frequency")
				.groupBy("UserId", "UserId_Index").count().drop("count");

		userIdTable.show();

		// <UserId,UserIndex,features(array)>
		Dataset<Row> userTableInfo = userIdTable
				.join(userALSFeatures, userIdTable.col("UserId_Index").equalTo(userALSFeatures.col("id"))).drop("id");
		userTableInfo.show();
		
		
		
		Dataset<Row> metadata_read = sparkSession.read().option("inferschema",
		 true).csv("data/metaData/*").toDF("SongId", "ArtistId");
		// Reading Data from a CSV file //Inferring Schema and Setting Header as True
		 
		Dataset<Row> metadata_read1 = metadata_read.na().drop();
		 metadata_read1.show();

		 
		 
		/*
		 * Dataset<Row> user_song_artist_table = metadata_read1.join(metadata_read1,
		 * user_click_stream_activity.col("SongId").equalTo(metadata_read1.col("SongId")
		 * ), "inner");
		 * 
		 * user_song_artist_table.show();
		 */
		
		 // metadata_read.join(metadata_read,modelIndexed("SongId_Index")==
		// metadata_read("SongId"));

		// Dataset<Row> user_song_artist_table =
		// userIdTable.join(userALSFeatures,
		// userIdTable.col("UserIndex").equalTo(userALSFeatures.col("id"))).drop("id");

		// userTableInfo.show();

		/*
		 * // Build the recommendation model using ALS ALS als = new
		 * ALS().setRank(10).setMaxIter(5).setRegParam(0.01).setUserCol("UserId_Index")
		 * .setItemCol("SongId_Index").setRatingCol("Frequency");
		 * 
		 * // Dataset<Row> songIndexed = indexer.fit(userIndexed
		 * ).transform(userIndexed);
		 * 
		 * Dataset<Row> modelIndexed = songIndexed .withColumn("UserIndex",
		 * col("UserId_Index").cast(DataTypes.IntegerType)) .withColumn("SongIndex",
		 * col("SongId_Index").cast(DataTypes.IntegerType));
		 * 
		 * 
		 * ALSModel model = als.fit(modelIndexed);
		 * 
		 * // Get the userFactors from ALS model to use it in kmeans Dataset<Row>
		 * userALSFeatures = model.userFactors(); userALSFeatures.show();
		 * 
		 * 
		 * Dataset<Row> userId_Index_Table = modelIndexed.drop("SongId_Index", "SongId",
		 * "Frequency") .groupBy("UserId", "UserId_Index").count();
		 * 
		 * userId_Index_Table.show();
		 * 
		 * // <UserId,UserIndex,features(array)> Dataset<Row> userTableInfo =
		 * userId_Index_Table.join(userALSFeatures,
		 * userId_Index_Table.col("UserId_Index").equalTo(userALSFeatures.col("id"))).
		 * drop("id");
		 * 
		 * userTableInfo.show(); userTableInfo.printSchema();
		 */ }
}

// df1.show(); //Displaying samples
// df1.printSchema(); //Printing Schema
// df1.describe().show(); //Statistically summarizing about the data

/*
 * //****************************************** Handling missing values
 * ******************************************************************
 * 
 * //Casting MPG and HORSEPOWER from String to Double Dataset<Row> df2 =
 * df1.select(col("MPG").cast("Double"), col("CYLINDERS"),col("DISPLACEMENT"),
 * col("HORSEPOWER").cast("Double"),col("WEIGHT"),
 * col("ACCELERATION"),col("MODELYEAR"),col("NAME"));
 * 
 * System.out.
 * println("*************************Casting columns********************************"
 * ); df2.show(); //Displaying samples df2.printSchema(); //Printing new Schema
 * 
 * //Removing Rows with missing values System.out.
 * println("********************Removing records with missing values**********************"
 * ); Dataset<Row> df3 = df2.na().drop(); //Dataframe.na.drop removes any row
 * with a NULL value df3.describe().show(); //Describing DataFrame
 * 
 * //******************************************Replace missing values with
 * approximate mean values*************************************
 * 
 * System.out.
 * println("*******************Replacing records with missing values********************"
 * );
 * 
 * //Imputer method automatically replaces null values with mean values. Imputer
 * imputer = new Imputer() .setInputCols(new String[]{"MPG","HORSEPOWER"})
 * .setOutputCols(new String[]{"MPG-Out","HORSEPOWER-Out"});
 * 
 * ImputerModel imputeModel = imputer.fit(df2); //Fitting DataFrame into a model
 * Dataset<Row> df4=imputeModel.transform(df2); //Transforming the DataFrame
 * df4.show(); df4.describe().show(); //Describing the dataframe
 * 
 * 
 * //Removing unnecessary columns Dataset<Row> df5 =df4.drop(new String[]
 * {"MPG","HORSEPOWER"});
 * 
 * //*******************************************Statistical Data
 * Analysis*************************************************************
 * 
 * System.out.
 * println("***********************Performing statistical exploration*********************"
 * );
 * 
 * StructType autoSchema = df5.schema(); //Inferring Schema
 * 
 * for ( StructField field : autoSchema.fields() ) { //Running through each
 * column and performing Correlation Analysis if ( !
 * field.dataType().equals(DataTypes.StringType)) { System.out.println(
 * "Correlation between MPG-Out and " + field.name() + " = " +
 * df5.stat().corr("MPG-Out", field.name()) ); } }
 * 
 * 
 * //****************************************Assembling the Vector and
 * Label************************************************************
 * 
 * System.out.
 * println("******************************Assembling the vector************************"
 * ); //Renaming MPG-Out as lablel Dataset<Row> df6=
 * df5.select(col("MPG-Out").as("label"),col("CYLINDERS"),col("WEIGHT"),col(
 * "HORSEPOWER-Out"),col("DISPLACEMENT"));
 * 
 * //Assembling the features in the dataFrame as Dense Vector VectorAssembler
 * assembler = new VectorAssembler() .setInputCols(new
 * String[]{"CYLINDERS","WEIGHT","HORSEPOWER-Out","DISPLACEMENT"})
 * .setOutputCol("features");
 * 
 * Dataset<Row> df7 = assembler.transform(df6).select("label","features");
 * df7.show();
 * 
 * //*********************************************Scaling the
 * Vector***********************************************************************
 * 
 * 
 * //Scaling the features between 0-1 MaxAbsScaler scaler = new MaxAbsScaler()
 * //Performing MaxAbsScaler() Transformation .setInputCol("features")
 * .setOutputCol("scaledFeatures");
 * 
 * // Building and Fitting in a MaxAbsScaler Model MaxAbsScalerModel scalerModel
 * = scaler.fit(df7);
 * 
 * // Re-scale each feature to range [0, 1]. Dataset<Row> scaledData =
 * scalerModel.transform(df7);
 * 
 * //*********************************************Normalizing the
 * Vector*********************************************************************
 * 
 * System.out.
 * println("**********************************Scaling and Normalizing the vector***************************"
 * ); //Normalizing the vector. Converts vector to a unit vector Normalizer
 * normalizer = new Normalizer() //Performing Normalizer() Transformation
 * .setInputCol("scaledFeatures") .setOutputCol("normFeatures") .setP(2.0);
 * 
 * Dataset<Row> NormData = normalizer.transform(scaledData); NormData.show();
 * 
 */
//}
//}
