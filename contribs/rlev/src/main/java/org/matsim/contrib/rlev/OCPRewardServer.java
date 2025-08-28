package org.matsim.contrib.rlev;

import com.google.common.util.concurrent.AtomicDouble;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.jfree.data.json.impl.JSONObject;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigReader;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.w3c.dom.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.*;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class OCPRewardServer {

    private final BlockingQueue<RequestData> requestQueue = new LinkedBlockingQueue<>();
    private final int threadPoolSize;
    private final ExecutorService executorService;

    private final AtomicDouble bestReward = new AtomicDouble(Double.NEGATIVE_INFINITY);
    private final AtomicBoolean initialResponse = new AtomicBoolean(true);

    public OCPRewardServer(int threadPoolSize){
        this.threadPoolSize = threadPoolSize;
        this.executorService = Executors.newFixedThreadPool(this.threadPoolSize);
    }

    public static void main(String[] args) throws Exception {
        int argsThreadPoolSize = Integer.parseInt(args[0]);
        OCPRewardServer rewardServer = new OCPRewardServer(argsThreadPoolSize);

        int port = 8000;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        System.setProperty("matsim.preferLocalDtds", "true");
        server.createContext("/getReward", rewardServer.new RewardHandler());
        server.setExecutor(null);
        System.out.println("Starting reward server...");
        server.start();
        System.out.println("Reward server is running on https://localhost:" + port);

        for (int i = 0; i < rewardServer.threadPoolSize; i++) {
            System.out.println("Starting thread: " + i);
            rewardServer.executorService.submit(rewardServer::processRequest);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                System.out.println("Shutting down server...");
                stopServer(server, rewardServer);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }));
    }

    public synchronized void setBestReward(double newReward){ bestReward.set(newReward); }
    public synchronized double getBestReward(){ return bestReward.get(); }

    private static void stopServer(HttpServer server, OCPRewardServer rewardServer) throws IOException {
        server.stop(0);
        rewardServer.executorService.shutdown();
        System.out.println("Server shut down gracefully.");
    }

    public void processRequest() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                System.out.println(Thread.currentThread().getName() + " Waiting for request...");
                RequestData data = this.requestQueue.take();
                HttpExchange exchange = data.getExchange();
                Path configPath = data.getFilePath();
                System.out.println("Processing request for config file: " + configPath + " with thread: " + Thread.currentThread().getName());

                File logFile = new File(configPath.getParent().toString(), "log.txt");

                // Load config and discover vehicles file name (may be null)
                Config config = new Config();
                new ConfigReader(config).parse(configPath.toUri().toURL());
                String vehiclesFileName = config.getParam("vehicles", "vehiclesFile");

                // Run MATSim in-process
                int exitCode = 0;
                try (PrintWriter log = new PrintWriter(new FileWriter(logFile, true))) {
                    log.println("=== Starting MATSim Controler.run() ===");
                    log.flush();
                    Config runCfg = ConfigUtils.loadConfig(configPath.toString());
                    Controler controler = new Controler(runCfg);
                    controler.run();
                    log.println("=== MATSim run completed ===");
                } catch (Throwable t) {
                    try (PrintWriter log = new PrintWriter(new FileWriter(logFile, true))) {
                        log.println("MATSim run failed:");
                        t.printStackTrace(log);
                    }
                    exitCode = 1;
                }
                System.out.println("Process exited with code: " + exitCode);

                // Compute rewards (robust to missing files)
                double chargeReward = 0.0;
                double timeReward = 0.0;

                Path outDir = configPath.getParent().resolve("output");
                Path chargeCsv = outDir.resolve("ITERS/it.0/0.average_charge_time_profiles.txt");
                Path legDurTxt = outDir.resolve("ITERS/it.0/0.legdurations.txt");
                Path evehiclesXMLPath = (vehiclesFileName == null || vehiclesFileName.isBlank())
                        ? configPath.getParent().resolve("vehicles.xml")
                        : configPath.getParent().resolve(vehiclesFileName);

                // Charge reward
                if (Files.exists(chargeCsv) && Files.isRegularFile(chargeCsv)) {
                    double avgEnergyCapacity = getAverageEnergyCapacity(evehiclesXMLPath.toString());
                    double avgChargeIntegral = 0.0;
                    double totRecords = 0.0;

                    try (Reader reader = new FileReader(chargeCsv.toString())) {
                        Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
                        for (CSVRecord record : records) {
                            String[] vals = record.values()[0].split("\t");
                            if (vals.length >= 3) {
                                double avgVal = Double.parseDouble(vals[2]);
                                avgChargeIntegral += avgVal;
                                totRecords += 1.0;
                            }
                        }
                    } catch (Exception e) {
                        try (PrintWriter log = new PrintWriter(new FileWriter(logFile, true))) {
                            log.println("Failed reading charge profiles: " + e.getMessage());
                        }
                    }

                    double totEnergyCapacity = avgEnergyCapacity * totRecords;
                    if (totEnergyCapacity > 0) chargeReward = avgChargeIntegral / totEnergyCapacity;
                } else {
                    try (PrintWriter log = new PrintWriter(new FileWriter(logFile, true))) {
                        log.println("Charge profiles file not found: " + chargeCsv);
                    }
                }

                // Time reward
                if (Files.exists(legDurTxt) && Files.isRegularFile(legDurTxt)) {
                    try {
                        String text = Files.readString(legDurTxt);
                        Matcher m = Pattern.compile("average leg duration:\\s+([0-9.]+)\\s+seconds").matcher(text);
                        if (m.find()) {
                            double seconds = Double.parseDouble(m.group(1));
                            timeReward = seconds / 86400.0; // normalize by seconds in a day
                        }
                    } catch (Exception e) {
                        try (PrintWriter log = new PrintWriter(new FileWriter(logFile, true))) {
                            log.println("Failed reading leg durations: " + e.getMessage());
                        }
                    }
                } else {
                    try (PrintWriter log = new PrintWriter(new FileWriter(logFile, true))) {
                        log.println("Leg durations file not found: " + legDurTxt);
                    }
                }

                JSONObject response = new JSONObject();
                response.put("filetype", initialResponse.get() ? "initialoutput" : "output");
                response.put("charge_reward", Double.toString(chargeReward));
                response.put("time_reward", Double.toString(timeReward));
                initialResponse.set(false);

                // Zip (create empty zip if output missing so the HTTP flow still completes)
                File outputFolder = outDir.toFile();
                File zipFile = configPath.getParent().resolve("output.zip").toFile();
                if (!outputFolder.exists()) {
                    outputFolder.mkdirs();
                }
                zipDirectory(outputFolder, zipFile);

                // Send response
                exchange.getResponseHeaders().set("X-Response-Message", response.toString());
                exchange.sendResponseHeaders(200, zipFile.length());

                try (OutputStream os = exchange.getResponseBody();
                     InputStream is = new FileInputStream(zipFile)) {
                    byte[] buffer = new byte[8192];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                    }
                }

                // cleanup temp folder
                for (int i = 0; i < 5; i++) {
                    try {
                        FileUtils.deleteDirectory(outputDir);
                        break;
                    } catch (IOException e) {
                        try { Thread.sleep(200L); } catch (InterruptedException ignored) {}
                        if (i == 4) throw e; // give up after 5 tries
                    }
                }
                System.out.println("Folder and subdirectories deleted successfully.");

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private static void zipDirectory(File folder, File zipFile) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(zipFile);
             ZipOutputStream zos = new ZipOutputStream(fos)) {
            zipFileRecursive(folder, folder.getName(), zos);
        }
    }

    private static void zipFileRecursive(File file, String entryName, ZipOutputStream zos) throws IOException {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] list = file.listFiles();
            if (list != null) {
                for (File sub : list) {
                    zipFileRecursive(sub, entryName + "/" + sub.getName(), zos);
                }
            }
        } else {
            zos.putNextEntry(new ZipEntry(entryName));
            try (FileInputStream fis = new FileInputStream(file)) {
                byte[] bytes = new byte[8192];
                int length;
                while ((length = fis.read(bytes)) >= 0) {
                    zos.write(bytes, 0, length);
                }
            }
            zos.closeEntry();
        }
    }

    public static double getAverageEnergyCapacity(String filePath) {
        try {
            File xmlFile = new File(filePath);
            if (!xmlFile.exists()) return 0.0;

            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(xmlFile);
            document.getDocumentElement().normalize();

            NodeList vehicleTypes = document.getElementsByTagName("vehicleType");
            List<Double> energyCaps = new ArrayList<>();

            for (int i = 0; i < vehicleTypes.getLength(); i++) {
                Node node = vehicleTypes.item(i);
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element vehicleType = (Element) node;
                    NodeList attributes = vehicleType.getElementsByTagName("attribute");
                    for (int j = 0; j < attributes.getLength(); j++) {
                        Element attribute = (Element) attributes.item(j);
                        if ("energyCapacityInKWhOrLiters".equals(attribute.getAttribute("name"))) {
                            String txt = attribute.getTextContent();
                            if (txt != null && !txt.isBlank()) {
                                double cap = Double.parseDouble(txt.trim());
                                energyCaps.add(cap);
                            }
                        }
                    }
                }
            }
            return energyCaps.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        } catch (Exception e) {
            e.printStackTrace();
            return 0.0;
        }
    }

    public class RewardHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String contentType = exchange.getRequestHeaders().getFirst("Content-Type");
            if (contentType == null || !contentType.contains("multipart/form-data")) {
                exchange.sendResponseHeaders(400, -1);
                return;
            }

            // boundary may be like: multipart/form-data; boundary=----WebKitFormBoundaryXYZ
            String[] ctParts = contentType.split("boundary=");
            if (ctParts.length < 2) {
                exchange.sendResponseHeaders(400, -1);
                return;
            }
            String boundary = ctParts[1];
            if (!boundary.startsWith("--")) boundary = "--" + boundary;

            // Read body
            ByteArrayOutputStream bodyOutput = new ByteArrayOutputStream();
            try (InputStream inputStream = exchange.getRequestBody()) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    bodyOutput.write(buffer, 0, bytesRead);
                }
            }
            String bodyString = bodyOutput.toString(StandardCharsets.UTF_8);

            // Temp folder
            String folderString = Long.toString(System.nanoTime());
            Path folderPath = Paths.get(System.getProperty("java.io.tmpdir"), folderString);
            Files.createDirectories(folderPath);

            // Parse parts (very simple multipart parsing)
            Path configPath = null;
            String[] parts = bodyString.split(boundary);
            for (String part : parts) {
                if (part.contains("Content-Disposition")) {
                    String[] lines = part.split("\r\n");
                    String fileName = extractFileName(lines);
                    if (fileName != null && !fileName.isBlank()) {
                        if (fileName.contains("config")) {
                            configPath = folderPath.resolve(fileName);
                        }
                        byte[] fileContent = extractFileContent(part);
                        saveFile(folderPath, fileName, fileContent);
                    }
                }
            }

            if (configPath == null) {
                exchange.sendResponseHeaders(400, -1);
                return;
            }

            System.out.println("Adding request for config file: " + configPath + " to queue for processing");

            try {
                // Force output dir into the temp folder so each request is isolated
                URL url = configPath.toUri().toURL();
                Config config = new Config();
                new ConfigReader(config).parse(url);
                config.setParam("controller", "outputDirectory", folderPath.resolve("output").toString());
                ConfigUtils.writeConfig(config, configPath.toString());

                requestQueue.add(new RequestData(exchange, configPath));
            } catch (Exception e) {
                e.printStackTrace();
                exchange.sendResponseHeaders(500, 0);
                exchange.getResponseBody().close();
            }
        }
    }

    private void saveFile(Path folderPath, String fileName, byte[] fileContent) throws IOException {
        Files.write(folderPath.resolve(fileName), fileContent);
    }

    private String extractFileName(String[] lines) {
        for (String line : lines) {
            if (line.contains("filename")) {
                String[] parts = line.split(";");
                for (String p : parts) {
                    p = p.trim();
                    if (p.startsWith("filename=")) {
                        return p.substring("filename=".length()).replace("\"","").trim();
                    }
                }
            }
        }
        return null;
    }

    private byte[] extractFileContent(String part) {
        int startIndex = part.indexOf("\r\n\r\n");
        int endIndex = part.lastIndexOf("\r\n--");
        if (startIndex < 0) return new byte[0];
        if (endIndex < 0) endIndex = part.length();
        return part.substring(startIndex + 4, endIndex).getBytes(StandardCharsets.UTF_8);
    }

    private static class RequestData {
        private final HttpExchange exchange;
        private final Path filePath;

        public RequestData(HttpExchange exchange, Path filePath) {
            this.exchange = exchange;
            this.filePath = filePath;
        }
        public HttpExchange getExchange() { return exchange; }
        public Path getFilePath() { return filePath; }
    }
}
