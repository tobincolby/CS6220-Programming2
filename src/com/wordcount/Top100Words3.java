package com.wordcount;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

public class Top100Words3 {

    public static final String TEMP_OUT_DIRECTORY = "/input/wordcount/indicies";

    static long mapTime = 0;
    static long reduceTime = 0;

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, Text>{

        private Text fileText = new Text();
        private Text word = new Text();
        private long mapStartTime;

        public void setup(Context context) {
            mapStartTime = System.currentTimeMillis();
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String fileName = ((FileSplit)(context.getInputSplit())).getPath().getName();
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                fileText.set(fileName);
                context.write(word, fileText);
            }
        }

        public void cleanup(Context context) {
            mapTime += System.currentTimeMillis() - mapStartTime;
        }
    }

    public static class FileArrayReducer
            extends Reducer<Text,Text,Text,IntWritable> {

        private Map<String, Integer> map = new HashMap<>();
        private long reduceStartTime;

        public void setup(Context context) {
            reduceStartTime = System.currentTimeMillis();
        }

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            Set<String> vals = new HashSet<>();
            for (Text value : values) {
                vals.add(value.toString());
            }


            map.put(key.toString(), vals.size());

        }

        @Override
        public void cleanup(Context context) throws IOException,
                InterruptedException
        {
            for (Map.Entry<String, Integer> entry : map.entrySet())
            {
                Text text = new Text();
                text.set(entry.getKey());
                context.write(text, new IntWritable(entry.getValue()));

            }
            reduceTime += System.currentTimeMillis() - reduceStartTime;
        }

    }

    public static class FileArrayMapper
            extends Mapper<Text, IntWritable, IntWritable, Text> {

        private long mapStartTime;

        public void setup(Context context) {
            mapStartTime = System.currentTimeMillis();
        }

        public void map(Text key, IntWritable value, Context context) throws IOException, InterruptedException {
            context.write(value, key);
        }

        public void cleanup(Context context) {
            mapTime += System.currentTimeMillis() - mapStartTime;
        }

    }

    public static class WordAppearanceReducer
            extends Reducer<IntWritable, Text, Text, IntWritable> {

        private TreeMap<Integer, ArrayList<String>> map = new TreeMap<>();
        private int count = 0;
        private long reduceStartTime;

        public void setup(Context context) {
            reduceStartTime = System.currentTimeMillis();
        }

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            ArrayList<String> vals = new ArrayList<>();
            for (Text value : values)
                vals.add(value.toString());

            map.put(key.get(), vals);

            count += vals.size();
            while (count > 100) {
                map.get(map.firstKey()).remove(0);
                if (map.get(map.firstKey()).isEmpty()) {
                    map.remove(map.firstKey());
                }
                count --;
            }
        }

        @Override
        public void cleanup(Context context) throws IOException,
                InterruptedException
        {
            for (Map.Entry<Integer, ArrayList<String>> entry : map.entrySet())
            {

                int count = entry.getKey();
                for (String word : entry.getValue()) {
                    context.write(new Text(word), new IntWritable(count));
                }

            }
            reduceTime += System.currentTimeMillis() - reduceStartTime;
        }

    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Top 100 Words File Count");
        job.setJarByClass(Top100Words.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(FileArrayReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        SequenceFileOutputFormat.setOutputPath(job, new Path(TEMP_OUT_DIRECTORY));
        long startTime = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        if (!success) {
            System.exit(1);
        }
        Job job2 = Job.getInstance(conf, "Top 100 Words Inverted Index");
        job2.setJarByClass(Top100Words.class);
        job2.setInputFormatClass(SequenceFileInputFormat.class);
        job2.setMapperClass(FileArrayMapper.class);
        job2.setReducerClass(WordAppearanceReducer.class);
        job2.setMapOutputKeyClass(IntWritable.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);
        SequenceFileInputFormat.addInputPath(job2, new Path(TEMP_OUT_DIRECTORY + "/part-r-00000"));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));
        success = job2.waitForCompletion(true);
        long stopTime = System.currentTimeMillis();
        System.out.println(startTime);
        System.out.println(stopTime);
        System.out.println("Run Time: " + (stopTime - startTime) + " ms");
        System.out.println("Map Time: " + mapTime + " ms");
        System.out.println("Reduce Time: " + reduceTime + " ms");
        System.exit(success ? 0 : 1);
    }
}

