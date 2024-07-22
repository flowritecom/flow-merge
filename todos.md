
## TODO

- [x] Resolve the model loading using internet
- [x] Finish running merge.plan() with the dummy snapshot 


- [ ] Figure out the RAM efficiency of EnrichedSnapshot

- [ ] Start building snapshot inside the merge.plan OR injected global state
  - [ ] Better defaults for the snapshot 
  - [ ] Check that the is_base is implemented and makes sense

- [ ] Create a wrapper function that registers the services as defaults
- [ ] Another that takes load +  plan + run (?)



```haskell

-- avoid this
-> "HJK"
  -> "XYZ"
  -> "XYZ"
  -> "XYZ"
  -> "XYZ"

-- do this
-> "XYZ"
-> "XYZ"
-> "XYZ"
-> "XYZ"
  -> "HJK"



-> "FileListValidator"

  

```
