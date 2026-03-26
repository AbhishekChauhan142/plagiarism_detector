## GitHub Copilot Chat

- Extension Version: 0.22.4 (prod)
- VS Code: vscode/1.95.0
- OS: Windows

## Network

User Settings:
```json
  "github.copilot.advanced": {
    "debug.useElectronFetcher": true,
    "debug.useNodeFetcher": false
  }
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.207.73.85 (16 ms)
- DNS ipv6 Lookup: Error: getaddrinfo ENOTFOUND api.github.com
- Electron Fetcher (configured): HTTP 200 (119 ms)
- Node Fetcher: HTTP 200 (122 ms)
- Helix Fetcher: HTTP 200 (161 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.114.21 (3 ms)
- DNS ipv6 Lookup: Error: getaddrinfo ENOTFOUND api.individual.githubcopilot.com
- Electron Fetcher (configured): HTTP 200 (298 ms)
- Node Fetcher: HTTP 200 (811 ms)
- Helix Fetcher: HTTP 200 (808 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).