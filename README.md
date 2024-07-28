# droplet-analysis

## Build a new docker image

```
docker build deploy/ -t droplet-analysis
docker run -v ~/droplet-analysis:/app -it droplet-analysis bash
```