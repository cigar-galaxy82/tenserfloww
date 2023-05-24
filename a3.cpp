#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

int serial_min(vector<int> arr)
{
	int n = arr.size();
	int minm = INT_MAX;
	for(int i=0; i<n; i++)
		minm = min(minm, arr[i]);
	return minm;
}

int parallel_min(vector<int> arr)
{
	int n = arr.size();
	int minm = INT_MAX;
	#pragma omp parallel for reduction(min:minm)
	for(int i=0; i<n; i++)
		minm = min(minm, arr[i]);
	return minm;
}

int serial_max(vector<int> arr)
{
	int n = arr.size();
	int maxm = INT_MIN;
	for(int i=0; i<n; i++)
		maxm = max(maxm, arr[i]);
	return maxm;
}

int parallel_max(vector<int> arr)
{
	int n = arr.size();
	int maxm = INT_MIN;
	#pragma omp parallel for reduction(max:maxm)
	for(int i=0; i<n; i++)
		maxm = max(maxm, arr[i]);
	return maxm;
}

int serial_sum(vector<int> arr)
{
	int n = arr.size();
	int sum=0;
	for(int i=0; i<n; i++)
		sum += arr[i];
	return sum;
}

int parallel_sum(vector<int> arr)
{
	int n = arr.size();
	int sum=0;
	#pragma omp parallel for reduction(+:sum)
	for(int i=0; i<n; i++)
		sum += arr[i];
	return sum;
}

int main()
{
	vector<int> arr;
	int n;
	cout<<"\n Enter the size of the array : ";
	cin>>n;
	
	for(int i=0; i<n; i++)
		arr.push_back(rand()%100);
		
	double start, end;
	float avg=0.0;
	
	start = omp_get_wtime();
		int minm = serial_min(arr);
		int maxm = serial_max(arr);
		int sum = serial_sum(arr);
		avg = (float)sum/n;
	end = omp_get_wtime();
	
	cout<<"\n*************** SERIAL EXECUTION ***************\n";
	cout<<"\n\t Minimum Element is : "<<minm;
	cout<<"\n\t Maximum Element is : "<<maxm;
	cout<<"\n\t Sum of the elements is : "<<sum;
	cout<<"\n\t Average is : "<<avg;
	cout<<"\n\t Time taken : "<<end-start<<" seconds "<<endl;
	
	
	start = omp_get_wtime();
		minm = parallel_min(arr);
		maxm = parallel_max(arr);
		sum = parallel_sum(arr);
		avg = (float)sum/n;
	end = omp_get_wtime();
	
	cout<<"\n*************** PARALLEL EXECUTION ***************\n";
	cout<<"\n\t Minimum Element is : "<<minm;
	cout<<"\n\t Maximum Element is : "<<maxm;
	cout<<"\n\t Sum of the elements is : "<<sum;
	cout<<"\n\t Average is : "<<avg;
	cout<<"\n\t Time taken : "<<end-start<<" seconds \n"<<endl;
	
	return 0;
}