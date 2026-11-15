# Local experiment viewer

Run `python3 viewer/server.py` and open http://127.0.0.1:8765.

The viewer reads saved receipts and original agent trajectories from the four experiment directories listed in `server.py`. Refresh rereads local files; it does not contact Valkyrie, run jobs, or alter results. The 45-minute comparison and 10-minute smoke tests remain separate. Scores are the saved verifier reward multiplied by 100, averaged only over scored, finished trials. Missing and failed runs are not zeros.

The server binds only to localhost, exposes a small allowlist of routes, and renders trace contents as text. It never reads private holdout inputs. Original hidden reasoning is omitted; recorded public messages, calls, and results retain their order.

Run the parser and aggregation checks with `python3 -m unittest viewer.test_server`.
