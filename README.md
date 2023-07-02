# droplet-analysis

```
docker build . -t droplet-analysis
docker run -it -p 8888:8888 -v .:/app --entrypoint bash droplet-analysis

jupyter notebook --allow-root --ip 0.0.0.0
```