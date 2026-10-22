<img width="660" height="133" alt="Screenshot 2026-10-21 183558" src="https://github.com/user-attachments/assets/8904fadc-91cf-4a34-bb35-7cace0e0f248" /><img width="1274" height="302" alt="Screenshot 2026-10-21 181941" src="https://github.com/user-attachments/assets/9ccd98e3-dca8-402e-9be5-d6728812b191" /># Agentic-RAG-chatbot
Completely local agentic rag chatbot 
Uses a Grader node to classify the questions, reranker consisting of vector and bm25 search and a cache.


So this is my first project regarding Agentic RAG, its a RAG pipeline built on top of a arxiv research paper.
I am going to talk about the plus points and the flaws present in this pipeline.
First going with the flaws, 
The latency is too high right now for a RAG pipeline, it takes about 150-200 seconds for the pipeline to answer a single question on a cold run, to counter that i have used caching method to drop the latency, which showed that the time dropped to half its original time in the cold run or even better which was 5-10 times less than the cold run.
This can be seen through the videos presented below which are of the fastapi endpoints for this project.
https://github.com/user-attachments/assets/f63a5836-fd7c-4efc-bab9-5585cc46fe85
As seen in this video, this tells us two things about the pipeline
First:- The system is not hallucinating when it comes to giving answers
Second:- The caching method works and improved the latency

WHY I DIDN'T USE ANY OTHER METHOD THAN CACHING TO IMPROVE LATENCY?
So the reason for me choosing caching and not other methods such as tightening the rerank score or limiting the top k vector hits was that i didn't want to compromise the answer quality, because this pipeline is completely local in the first place so some issues arrive from there also and tightening the parameters on top of that could have resulted in poor answer quality, this is something i look forward to counter or improve in my future projects.

Then coming on the eval results, they came back very positive too
As seen there are vector hits and bm25 search for questions that are answerable and most of the time the rerank score is very close to the ideal range (majority being 0.8-1), though it is seen that the pipeline is not doing that well in extract type questions, something which i saw too when running the questions myself, will strive to improve this as well.

This whole project is written mostly by Claude except for the dataset used to evaluate the pipeline, i wrote that myself looking at the data on which this pipeline is based upon. 

SO IF MOST OF THE PROJECT IS WRITTEN BY CLAUDE THEN WHAT DID I DO?
1. Wrote the dataset.
2. Ran sanity checks after each function and for each script to ensure that the pipeline is working as expected.
3. Decided the structure of this project.
4. Fixed errors arising in the pipeline.
5. Ran evals and measured them.


**Here is the proof for me running Sanity checks for each script**
**Schemas.py**
[sanity_check_code_block.md](https://github.com/user-attachments/files/33826641/sanity_check_code_block.md), This is the code block for sanity checking schemas
<img width="1283" height="88" alt="Image" src="https://github.com/user-attachments/assets/e76474c4-917f-4fa6-9a28-233b80f53c56" />
Result for schemas' sanity check, (Note - U may question that in the picture the project is named legal_doc_assistance but here it is agent_rag_chatbot, that is because i initially chose a legal doc dataset for this project but then shifted to this and changed the name recently only.)

**Tools.py**
[retrieval_sanity_check.md](https://github.com/user-attachments/files/33826923/retrieval_sanity_check.md)
<img width="1274" height="302" alt="Image" src="https://github.com/user-attachments/assets/76fce51b-4c01-4b9f-8cb4-c7421ffc8c57" />

**Graph.py**
[graph_workflow_sanity_check.md](https://github.com/user-attachments/files/33827091/graph_workflow_sanity_check.md)
<img width="660" height="133" alt="Image" src="https://github.com/user-attachments/assets/9b434d7d-60b4-4740-a714-89d692e236b8" />****

For ingestion it was 
print (metadatas(0))
print (metadatas(50))
And the answer came back to 19.7

So that is it, for my next projects some key takeaways are improving the latency without trading off the answer quality, incorporate semantic and embedding cache in the workflow too, maybe improve the dataset by making it bigger as well.

Thankuu!!










