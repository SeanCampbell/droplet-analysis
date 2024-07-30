# droplet-analysis

## Build a new docker image

```bash
docker build deploy/ -t droplet-analysis
docker run -v ~/droplet-analysis:/app -it droplet-analysis bash
```

## References

https://github.com/SeanCampbell/droplet-analysis-test-02/blob/main/src/video-processor/droplet-detector/image.py