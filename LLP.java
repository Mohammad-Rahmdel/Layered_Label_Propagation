import java.util.*;
import java.io.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.Random;


public class LLP {
    public static float NMI(Vector<Integer> Prediction, String TrueCommunityPathTXT )throws IOException {
        Vector<Integer> TrueLabel = new Vector<Integer>();
        int countGuess = 0, countGold = 0;
        float NMI = 0, up = 0, down = 0;
        int n = 0;
        TrueLabel.add(0);
        BufferedReader br = new BufferedReader(new FileReader(TrueCommunityPathTXT));
        String line = br.readLine();
        while (line != null) {
            String[] parts = line.split(" ");
            TrueLabel.add(Integer.parseInt(parts[1]));
            n++;
            line = br.readLine();
        }
        br.close();
        if (n != Prediction.size() - 1)
            return -1;
        Hashtable<Integer, Integer> temp = new Hashtable<Integer, Integer>();
        int k = 1;
        for (int i = 1; i <= n; i++) {
            if (temp.containsKey((Integer) Prediction.get(i)))
                Prediction.set(i, temp.get(Prediction.get(i)));
            else {
                temp.put(Prediction.get(i), k);
                Prediction.set(i, temp.get(Prediction.get(i)));
                k++;
            }
        }
        for (int i = 1; i <= n; i++) {
            if (Prediction.get(i) > countGuess)
                countGuess = Prediction.get(i);
            if (TrueLabel.get(i) > countGold)
                countGold = TrueLabel.get(i);
        }
        float NRow[] = new float[countGold];
        float NCol[] = new float[countGuess];
        float matrix[][] = new float[countGold][countGuess];
        for (int i = 0; i < countGold; i++)
            matrix[i] = new float[countGuess];
        for (int i = 1; i <= n; i++) {
            matrix[TrueLabel.get(i) - 1][Prediction.get(i) - 1]++;
            NRow[TrueLabel.get(i) - 1]++;
            NCol[Prediction.get(i) - 1]++;
        }
        for (int i = 0; i < countGold; i++)
            if (NRow[i] != 0)
                down += NRow[i] * (Math.log(NRow[i] / (float) (n)))/Math.log(2);
        for (int i = 0; i < countGuess; i++)
            if (NCol[i] != 0)
                down += NCol[i] * (Math.log(NCol[i] / (float) (n)))/Math.log(2);
        for (int i = 0; i < countGold; i++)
            for (int j = 0; j < countGuess; j++)
                if (matrix[i][j] != 0)
                    up += matrix[i][j] * (Math.log((matrix[i][j] * (float) (n) / ((NRow[i] * NCol[j]))))/Math.log(2));
        up *= (float) -2;
        NMI = up / down;
        return NMI;
    }

    static void shuffleVector(Vector<Integer> v) {
        Random rnd = ThreadLocalRandom.current();
        for (int i = v.size() - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            int a = v.get(index);
            v.set(index,v.get(i));
            v.set(i,a);
        }
    }

    static void shuffleArray(int[] ar) {
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    static void read_input(Vector graph,int n) throws IOException{
        for(int i=0; i<n; i++)
            graph.add(new Vector<Integer>());
        BufferedReader br = new BufferedReader(new FileReader("network.txt"));
        String line = br.readLine();
        while (line != null) {
            String[] parts = line.split(" "); // reading each line
            int source = Integer.parseInt(parts[0]); // first number
            int destination = Integer.parseInt(parts[1]); // second number
            ((Vector) (graph.get(source-1))).add(destination);
            line = br.readLine();
        }
        br.close();
    }

    static void show_number_of_groups_at_the_end(int labels[]){
        int groups = 0;
        for (int x:labels)
            if (x!=0)
                groups++;
        System.out.println("Number of groups :" + groups);
    }

    static void show(int n ,long startTime ,long endTime , int nodes_label[] , int labels[]) throws IOException{
        Vector result = new Vector();
        result.add(0); // vector starts from 1
        for (int i=0; i<n; i++)
            result.add(nodes_label[i]);

        System.out.println("Total execution time: " + (endTime - startTime) + "ms");

        float pd = NMI(result,System.getProperty("user.dir")+ "\\community.txt"); // nmi = pd
        System.out.println("NMI : " + pd);
        //show_number_of_groups_at_the_end(labels);
    }

    static void layered_label_propagation(int n, int nodes_label[],int labels[], double gamma[], Vector graph , int enough[], int end_num){
        int[] pi = new int[n]; // for choosing nodes randomly
        for(int i=0; i<pi.length; i++)
            pi[i] = i+1; // 1 to n values

        boolean end = false; // controls ending conditions
        int print = 0; // number of algorithm iterations
        shuffleArray(pi); // Shuffling the nodes
        while (end==false) {
            print++;
            int cnt = 0;
            for (int i = 0; i < pi.length; i++) {
                int node = pi[i];
                int old_label = nodes_label[node - 1]; // label of random selected node

                Random rand = ThreadLocalRandom.current();
                int random = rand.nextInt(i + 2); // choose a random number between 0 and i+1
                double gama = gamma[random]; // select random gamma

                Vector nodes_neighbours =  (Vector) graph.get(node - 1); //neighbours of node
                shuffleVector(nodes_neighbours);// shuffling the neighbours


                if (nodes_neighbours.size() != 0) {// if node has neighbour
                    HashMap<Integer, Integer> neighbours = new HashMap<Integer, Integer>();// first integer is label of the node
                    //and the second integer is it's occurrence.
                    neighbours.put(nodes_label[(Integer) nodes_neighbours.get(0) - 1], 1); // first neighbour. definitely it's occurrence is 1
                    for (int j = 1; j < nodes_neighbours.size(); j++) {
                        boolean key = neighbours.containsKey(nodes_label[(Integer) nodes_neighbours.get(j) - 1]);
                        if (key == false) { // if node is new
                            neighbours.put(nodes_label[(Integer) nodes_neighbours.get(j) - 1], 1);
                        }
                        else // if node occurance is more than 1.
                        {
                            int val = neighbours.get(nodes_label[(Integer) nodes_neighbours.get(j) - 1]);
                            val++; // increment the occurance
                            neighbours.put(nodes_label[(Integer) nodes_neighbours.get(j) - 1], val);
                        }
                    }
                    int new_label = -1;
                    double v_i;
                    double k_i;
                    double max = -1000; //

                    for (Map.Entry map : neighbours.entrySet()) { // calculate the equation for every label which is neighbour
                        int search = (Integer) map.getKey();
                        v_i = labels[search - 1];
                        k_i = neighbours.get(search);
                        double tmp = k_i - gama*(v_i - k_i); //Equation for LLP
                        //double tmp = k_i - gama*(v_i - k_i); // Equation for LPA
                        if (max < tmp) { // finding maximum
                            max = tmp;
                            new_label = search;
                        }
                    }

                    labels[old_label - 1]--;
                    labels[new_label - 1]++;
                    nodes_label[node - 1] = new_label; // updating info

                }
                else // when a node has not any neighbours.
                {
                    cnt++;
                }
            }
            // three ending conditions
            if(show_details(cnt,print,labels,end,enough) == true)
                end = true;
            if(print == end_num) // when algorithm run ten times
                end = true;
            if(cnt == n) // cnt == n when all labels don't change.
                end = true;
        }
    }

    static boolean show_details(int cnt,int print , int labels[] ,boolean end , int enough[]){
//        System.out.println("CNT = " + cnt);
//        System.out.println("iter = " + print);
        int gps = 0;
        for (int x:labels)
            if (x!=0)
                gps++;
        enough[0] = enough[1];
        enough[1] = gps;
        if (enough[0] != 0) {
            double en = (enough[0] - enough[1]);
            en /= enough[0];
            if (en < 0.01) {
                //System.out.println("Number of groups :" + gps);
                return true;
            }
        }
        //System.out.println("Number of groups :" + gps);
        return false;
    }

    static void create_result_txt(int nodes_label[] , int n) throws IOException{
        PrintWriter writer = new PrintWriter("result.txt", "UTF-8");
        for (int i=1; i<=n; i++)
            writer.println(i + " " + nodes_label[i-1]);
        writer.close();
    }

    static void make_gamma(double gamma[]){
        for (int i=1; i<gamma.length; i++)
            gamma[i] = 1/(Math.pow(2,i));
    }
    static void make_gamma2(double gamma[], int r){
        for (int i=1; i<r; i++)
            gamma[i] = 1/(Math.pow(2,i));

        for (int i=r; i<gamma.length; i++)
            gamma[i] = 1/(Math.pow(2,r));
    }
    static void make_gamma3(double gamma[], int r){
        for (int i=1; i<r; i++)
            gamma[i] = 1/(Math.pow(2,i));

        for(int i=r; i<gamma.length; i++)
            gamma[i] = gamma[i-r+1];
    }

    static void init(int nodes_label[],int labels[]){
        for (int i=0; i<nodes_label.length; i++)
            nodes_label[i] = (i+1);  // at first label of the nodes = #number of the nodes.
        for (int i=0; i<labels.length; i++)
            labels[i] = 1;
    }

    public static void main(String []args)throws IOException {
        System.out.println("*****************************");
        Vector graph = new Vector<Vector>();

        int n = 50000;// number of nodes.

        read_input(graph,n);// reading network.txt
        long startTime = System.currentTimeMillis();// start calculating time

        int enough[] = new int[2]; // enough : one of the ending conditions
        int []nodes_label = new int[n]; // label of each node.
        int []labels = new int[n]; // labels counter
        init(nodes_label,labels); // nodes_label[i] = i+1    labels[i] = 1
        double []gamma = new double[n+1]; //Gamma random value

        make_gamma(gamma);
        //make_gamma3(gamma,r) r=10,50,100

        layered_label_propagation(n,nodes_label,labels,gamma,graph,enough,10); //set end_num
        create_result_txt(nodes_label,n);

        long endTime = System.currentTimeMillis();
        show(n,startTime,endTime,nodes_label,labels);

        System.out.println("*****************************");
    }
}
