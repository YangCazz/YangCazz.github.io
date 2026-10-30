// Shares full-article discovery and strict assertions with the page regression suite.
import { runChecks } from './verify-pages.mjs';
runChecks({ diagramsOnly: true }).catch(error => {
    console.error(error.message);
    process.exitCode = 1;
});
