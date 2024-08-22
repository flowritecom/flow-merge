# Overview 

<code-block lang="mermaid">
graph
   A[Load config]
   A -- From file --> B[MergePlan]
   A -- Define object in code --> B[MergePlan]
   B -- Loop through normalized slices to merge --> C[Save output model files] 
   C --> D[Create output model config]
   D --> E[Upload to HF ?]
   E -- " " --> F[Finished!]
</code-block>