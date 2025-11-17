import asyncio, time
import aiohttp

URL = "http://127.0.0.1:8000/ask"
PAYLOAD = {"query": "如何重設密碼？"}

async def worker(session, n_requests):
    for _ in range(n_requests):
        async with session.post(URL, json=PAYLOAD) as resp:
            await resp.text()

async def main(concurrency=10, requests_per_worker=20):
    async with aiohttp.ClientSession() as session:
        tasks = [worker(session, requests_per_worker) for _ in range(concurrency)]
        start = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start
        total_requests = concurrency * requests_per_worker
        print(f"Total requests: {total_requests}")
        print(f"Total time: {duration:.2f}s")
        print(f"Requests per second: {total_requests/duration:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
