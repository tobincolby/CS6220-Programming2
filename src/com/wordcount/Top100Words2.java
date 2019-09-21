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

public class Top100Words2 {

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

        private Map<String, Set<String>> map = new HashMap<>();
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
            map.put(key.toString(), vals);

        }

        @Override
        public void cleanup(Context context) throws IOException,
                InterruptedException
        {

            int count = 0;
            TreeMap<Integer, ArrayList<String>> treeMap = new TreeMap<>();
            for (Map.Entry<String, Set<String>> entry : map.entrySet())
            {

                int numFiles = entry.getValue().size();
                String file = entry.getKey();

                if (treeMap.containsKey(numFiles)) {
                    treeMap.get(numFiles).add(file);
                } else {
                    ArrayList<String> arr = new ArrayList<>();
                    arr.add(file);
                    treeMap.put(numFiles, arr);
                }
                count += 1;
                while (count > 100) {
                    treeMap.get(treeMap.firstKey()).remove(0);
                    if (treeMap.get(treeMap.firstKey()).isEmpty()) {
                        treeMap.remove(treeMap.firstKey());
                    }
                    count --;
                }
            }


            for (Map.Entry<Integer, ArrayList<String>> entry : treeMap.entrySet())
            {

                int total = entry.getKey();
                for (String word : entry.getValue()) {
                    context.write(new Text(word), new IntWritable(total));
                }

            }

            reduceTime += System.currentTimeMillis() - reduceStartTime;
        }

    }



    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Top 100 Words File Count - 1 Job");
        job.setJarByClass(Top100Words2.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(FileArrayReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
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

