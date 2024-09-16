# Notes on things done
### Optomizations
1) Tested different quantity's of workers, found that 2 was the best for me as it is the first value that saturated my GPU
2) set pin_memory=True on dataloaders, this is an optomization for NVIDIA GPU's
3) tested different batch sizes. I found for my system that a batch size of 32 was fastest, 16 and 64 where also fast, but 32 was the top of the curve. smaller and larger batches ran less efficiently.
4) set non-blocking to true on GPU devices. this helped with a GPU error i would sometimes get.

### Loss functions
It appears that MSE is the best loss function available for this task. Cross entropy gave some psychodelic looking outputs but the results where not at all correct. MSE on the other hand while the outputs where muted the image at least looked better.
