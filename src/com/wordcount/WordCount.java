package com.wordcount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {


    static long mapTime = 0;

    static long reduceTime = 0;

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private long mapStartTime;

        public void setup(Context context) {
            mapStartTime = System.currentTimeMillis();
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }

        public void cleanup(Context context) {
            mapTime += System.currentTimeMillis() - mapStartTime;
        }
    }

    public static class IntSumReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();
        private long reduceStartTime;


        public void setup(Context context) {
            reduceStartTime = System.currentTimeMillis();
        }

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }

        public void cleanup(Context context) {
            reduceTime += System.currentTimeMillis() - reduceStartTime;
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        long startTime = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        long stopTime = System.currentTimeMillis();
        System.out.println(startTime);
        System.out.println(stopTime);
        System.out.println("Run Time: " + (stopTime - startTime) + " ms");
        System.out.println("Map Time: " + mapTime + " ms");
        System.out.println("Reduce Time: " + reduceTime + " ms");
        System.exit(success ? 0 : 1);
    }
}
