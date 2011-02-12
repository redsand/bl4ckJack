
#include "bl4ckjack.h"
#include "bl4ckJack_distributed.h"
#include "bl4ckJack_distributed_server.h"
#include "bl4ckJack_distributed_service.h"

// Client == GPU/CPU bruteforcing interface
// Server == GUI asking for updates and distributing pieces of keyspace
//				keyspace: { keyspace#: 1-32, token#: ( (charsetLen ^ keyspace) / (clients * token_factor) ),
//							double startToken, double endToken
//						 }

#include <math.h>
#include <stdio.h>
#include <Qt>
#include <QString>
#include <QDebug>

#include <RCF/Idl.hpp>
#include <RCF/IpServerTransport.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/TcpEndpoint.hpp>
#include <RCF/FilterService.hpp>
#include <RCF/ZlibCompressionFilter.hpp>
#include <RCF/OpenSslEncryptionFilter.hpp>

#include "bl4ckJack_bruteforce.h"


void bl4ckJackBrute::doWork() {

	// startup our brute thread

	QString status;
	// calculate our keyspace
	
	if(this->listenOnly) {

		distributedServer = new DistributedServer;
		QStringList l = this->listenHost.split(":");
		if(l.count() > 1)
			distributedServer->setArgs((char *)l[0].toAscii().constData(), l[1].toInt());
		else if (l.count() > 0)
			distributedServer->setArgs((char *)l[0].toAscii().constData());
		else
			distributedServer->setArgs();

		distributedServer->start();

		emit updateBruteStatus(3, 0, tr("Listening on %1 for distributed requests.").arg(this->listenHost));

		distributedServer->wait();
		distributedServer->terminate();
		return;
	}

	long double maxVal = 0;
	int keyLen=0;
	
	// break up our keyspace into array
	// each peice: (max size / ((serverCount) * tokensPerServer));
	// maxVal = charsetLen^ keyLen (16) + keyLen(15) etc
	keyLen = settings->value("config/dc_max_password_size").toInt();
	for(int i=1; i <= keyLen; i++) {
		maxVal += ((long double) pow((long double)settings->value("config/charset").toString().length(), i));
	}

	emit updateBruteStatus(3, 0, tr("Total passwords to calculate (%1x%2): %3").arg(settings->value("config/charset").toString().length()).arg(keyLen).arg((double)maxVal));

	// connect to list of hosts for bruting
	QStringList serverList = settings->value("config/dc_hosts").toStringList();
	QStringList successfulServerList;

	distributedServer = new DistributedServer;
	// start our local listener (if enabled) and add to list
	if(settings->value("config/dc_local_service", true).toBool()) {
		emit updateBruteStatus(3, 0, tr("Starting our local distributed agent..."));
		
		QStringList l = this->listenHost.split(":");
		if(l.count() > 1)
			distributedServer->setArgs((char *)l[0].toAscii().constData(), l[1].toInt());
		else if (l.count() > 0)
			distributedServer->setArgs((char *)l[0].toAscii().constData());
		else
			distributedServer->setArgs();

		distributedServer->start();

		if(!serverList.contains("127.0.0.1") && !serverList.contains("localhost"))
			serverList.append("127.0.0.1");
	} /* else
		qDebug() << "Configured not to start local distributed server"; */

	int i = 0;
	QList< RcfClient<RemoteService> *> serverConList;
	RcfClient<RemoteService> *client=NULL;
	for(i = 0; i < serverList.count(); i++) {

		//		send each 1 piece of keyspace
		//		if no response, server is pruned from list and key is passed to next
		
		int iter=0;
		emit updateBruteStatus(3, (float)((i+1) / serverList.count()) * 100, QString("Connecting to remote service %1").arg(serverList[i]));

		try {
			client = new RcfClient<RemoteService>( RCF::TcpEndpoint(serverList[i].toUtf8().constData(), DEFAULT_PORT) );
		} catch(const RCF::Exception &e) {
			emit updateBruteStatus(2, (float)((i+1) / serverList.count()) * 100, tr("Failed to connect to remote service %1,%2").arg(serverList[i]).arg(e.getErrorString().c_str()));
			continue;
		}

		client->getClientStub().setConnectTimeoutMs(settings->value("config/dc_timeout",2*1000).toInt());

		std::vector<RCF::FilterPtr> filters;
		filters.clear();

		if(settings->value("config/dc_compression", true).toBool()) {
			try {
				filters.push_back(RCF::FilterPtr( new RCF::ZlibStatefulCompressionFilter()));
				iter++;
			} catch(const RCF::Exception &e) {
				emit updateBruteStatus(2, (float)((i+1) / serverList.count()) * 100, QString("Failed to add zlib compression filter to remote service %1: %2").arg(serverList[i]).arg(e.getErrorString().c_str()));	
				continue;
			}
		}
		
		if(settings->value("config/dc_ssl_encryption", true).toBool()) {
			try {
				filters.push_back( RCF::FilterPtr( 
					new RCF::OpenSslEncryptionFilter(
					settings->value("config/dc_ssl_encryption_pem_file","path/to/serverCert.pem").toByteArray().constData(),
					settings->value("config/dc_ssl_encryption_pem_password","password").toByteArray().constData()
				)));
				iter++;
			} catch(const RCF::Exception &e) {
				emit updateBruteStatus(2, (float)((i+1) / serverList.count()) * 100, QString("Failed to add encryption filter to remote service %1, %2").arg(serverList[i]).arg(e.getErrorString().c_str()));	
				continue;
			}
		}

		try {
			if(iter > 0) {
				client->getClientStub().requestTransportFilters(filters);
			}
		} catch(const RCF::Exception &e) {
			emit updateBruteStatus(2, (float)((i+1) / serverList.count()) * 100, QString("Failed to add filters to remote service %1, %2").arg(serverList[i]).arg(e.getErrorString().c_str()));	
			continue;
		}

		try {
			std::string mymodule(this->EnabledModule.toAscii().constData());
			client->getClientStub().setRequestUserData(mymodule);
			client->initModule();
		} catch(const RCF::Exception &e) {
			emit updateBruteStatus(2, (float)((i+1) / serverList.count()) * 100, QString("Failed to initialize our module %1 with remote service %2, %3").arg(this->EnabledModule).arg(serverList[i]).arg(e.getErrorString().c_str()));	
			continue;
		}

		try {
			std::string mychars(settings->value("config/charset").toByteArray().constData());
			client->getClientStub().setRequestUserData(mychars);
			client->initKeyspace();
		} catch(const RCF::Exception &e) {
			emit updateBruteStatus(2, (float)((i+1) / serverList.count()) * 100, QString("Failed to initialize our keyspace with remote service %1, %2").arg(serverList[i]).arg(e.getErrorString().c_str()));	
			continue;
		}

		// if connected, append to successfulServerList
		if(client) {
			serverConList.append(client);
			successfulServerList.append(serverList[i]);
		} else continue;

	}

	long double pertoken = ceil(maxVal / (successfulServerList.count() * settings->value("config/dc_minimum_tokens",10).toInt()));

	double tokencount = floor(maxVal / pertoken);
	double tokeniter = 0;
	
	
	// if 0 available hosts, notify and stop bruteforce
	if(successfulServerList.count() > 0) {
		emit updateBruteStatus(3, 1, tr("Distributing %1 hashes to %2 hosts for processing...").arg(bJMain->tblHashView->getList().count()).arg(successfulServerList.count()));
	} else {
		emit updateBruteStatus(1, 1, tr("Failed to identify hosts for the hash distribution process."));
		return;
	}

	if(bJMain->tblHashView->getList().count() == 0) {
		emit updateBruteStatus(1, 1, tr("Failed to identify hsashes for the remote distribution process."));
		return;
	}

	for(i = 0; i < successfulServerList.count(); i++) {
		
		int j=0;
		for(j=0; j < bJMain->tblHashView->getList().count(); j++) {
			QPair<QString, QString> pair = bJMain->tblHashView->getList().at(j);
			
			try {
				if(this->EnabledModule.compare(pair.first) == 0) {
					std::string hash(pair.second.toAscii().constData());
					serverConList[i]->getClientStub().setRequestUserData(hash);
					serverConList[i]->initHash();
					if(j % 10 == 0)
					emit updateBruteStatus(3, (float)((j+1) / bJMain->tblHashView->getList().count()) * 100, tr("Distributing hash %1 to remote host %2.").arg(j).arg(serverList[i]));
				}
			} catch(const RCF::Exception &e) {
				emit updateBruteStatus(2, (float)((j+1) / bJMain->tblHashView->getList().count()) * 100, QString("Failed to initialize our hash %1 with remote service %2, %3").arg((char *)pair.second.toAscii().constData()).arg(serverList[i]).arg(e.getErrorString().c_str()));	
				continue;
			}
		}
		emit updateBruteStatus(3, 100, tr("Distributing %1 total hashes to remote host %2.").arg(j).arg(serverList[i]));
	}

	for(i = 0; i < successfulServerList.count(); i++) {
		
		emit updateBruteStatus(3, ((i+1) / successfulServerList.count()) * 100, tr("Distributing tokens #%1 to remote host %2.").arg(tokeniter).arg(successfulServerList[i]));
		try {
			if(serverConList[i]->submitKeyspace(tokeniter * pertoken, ((tokeniter + 1) * pertoken) - 1)) {
				tokeniter++;
			}
		} catch(const RCF::Exception &e) {
			emit updateBruteStatus(2, (float)((i+1) / successfulServerList.count()) * 100, tr("Failed to initialize our token #%1 with remote service %2, %3").arg(tokeniter).arg(serverList[i]).arg(e.getErrorString().c_str()));
			continue;
		}
	}

	// start our bruteforce for each

	for(i = 0; i < successfulServerList.count(); i++) {
		emit updateBruteStatus(3, (float)((i+1) / successfulServerList.count()) * 100, tr("Starting brute force with remote host %1.").arg(successfulServerList[i]));
		try {
			serverConList[i]->start();
		} catch(const RCF::Exception &e) {
			emit updateBruteStatus(2, (float)((i+1) / successfulServerList.count()) * 100, tr("Failed to start brute force with remote host %1: %2").arg(serverList[i]).arg(e.getErrorString().c_str()));
			continue;
		}
	}
	
	// listen for update/signal types:
	//		- password found
	//		- keyspace 95% completed
	//		- performance statistics

	// at 95% completion rate, w're expected to send the requesting host its next keyspace
	//		upon success crack, thread should add password to gui

	// sleep and let our stuff catch up
	/*
	emit updateBruteStatus(3, 0, tr("Bruteforcing waiting 5 seconds for nodes to catch up."));
	this->msleep(5000);
	*/
	
	qint64 totalHashFound = 0;

	int retry=1;

	qint64 hashFoundPrev = 0;
	double pct = 0.0;
	long double seconds_left = 0.0;
	int days = 0;
	int hours = 0;
	int minutes = 0;
	int seconds = 0;
	long double keysLeft = 1;

	while(!this->stop && keysLeft > 0) {
		while(this->go == false) {
			qDebug() << "paused.";
			emit updateBruteStatus(3, pct, tr("Bruteforcing paused."));
			msleep(500);
		}
//		once completed, wait for progress updates and send more keys upon completion

		// loop through serverConList and look for matched passwords

		BruteForceMatch match;
		for(i=0; i < serverConList.count(); i++) {
			while(!this->stop) {
				try {
					serverConList[i]->getMatch(match);
					if(match.hash.empty() || match.password.empty())
						break;
					emit updateBrutePassword(QString(match.hash.c_str()), QString(match.password.c_str()));
				} catch(const RCF::Exception &e) {
					emit updateBruteStatus(3, pct, tr("Failed to get potential matches from remote host %1: %2").arg(serverList[i]).arg(e.getErrorString().c_str()));
					continue;
				}
			}
		}

		// for each connection, we need to get the # of keys NOT done, per host
		// calc a final value of #'s to crack and then calc how long it will take
		// by using our pps input

		QString tmp;
		long double pps = 0;
		qint64 hashFoundPrev = 0;
		keysLeft = 0;
		for(i=0; i < serverConList.count(); i++) {
			BruteForceStats *s = new BruteForceStats;
			try {
				serverConList[i]->getStats(*s);
				pps += s->milHashSec;
				keysLeft += s->currentOpenTokens;
				totalHashFound += s->totalHashFound;
				delete s;
			} catch(const RCF::Exception &e) {
				emit updateBruteStatus(3, pct, tr("Failed to get stats from remote host %1: %2").arg(serverList[i]).arg(e.getErrorString().c_str()));
				continue;
			}

			//if(tokeniter*pertoken > maxVal) break;
			// need to get total keyspace left from hosts && from keyspaceList
			
			if(retry++ % 10 == 0) {
				for(i = 0; i < successfulServerList.count(); i++) {
			
					try {
						if(serverConList[i]->submitKeyspace(tokeniter * pertoken, ((tokeniter + 1) * pertoken) - 1)) {
							tokeniter++;
							//emit updateBruteStatus(3, pct, tr("Distributing tokens #%1 to remote host %2.").arg(tokeniter).arg(successfulServerList[i]));
							//usleep(1000);
						}
					} catch(const RCF::Exception &e) {
						//emit updateBruteStatus(2, pct, tr("Failed to initialize our token #%1 with remote service %2, %3").arg(tokeniter).arg(serverList[i]).arg(e.getErrorString().c_str()));
						continue;
					}
				}
				retry = 1;
			}

		}

		//qDebug() << "Adding " << (double)keysLeft << " with " << (double)(pertoken * ((tokencount - tokeniter)+1));
		keysLeft += (pertoken * ((tokencount - tokeniter)+1));
		// pps (a second)
		// keysLeft (total keys)

		if(pps > 0) {
			seconds_left = keysLeft / (pps * 1000000);
			//qDebug() << "Keysleft " << (double)keysLeft << " pps " << (double)(pps * 1000000) << " secLeft " << (double)seconds_left;
			days = (int) floor(seconds_left / 60 / 60 / 24);
			hours = (int) fmod(seconds_left / 60 / 60, 24);
			minutes = (int) fmod(seconds_left / 60, 60);
			seconds = (int) fmod(seconds_left, 60);
			//qDebug() << days << " days, " << hours << ":" << minutes << ":" << seconds;
			pct = ((maxVal - keysLeft) / maxVal) * 100.0;
			if(pct < 0)
				pct = 0;
			//qDebug() << " pct: << " << pct << " ( maxVal (" << (double)maxVal << ") - keysLeft(" << (double)keysLeft << ") / maxVal(" << (double)maxVal << ") ) * 100";	
		}

		if(pps > 0 || totalHashFound > hashFoundPrev) {
			hashFoundPrev = totalHashFound;
			emit updateBruteLabels(pps, tr("%1 days, %2:%3:%4").arg(days).arg(tmp.sprintf("%02d",hours)).arg(tmp.sprintf("%02d",minutes)).arg(tmp.sprintf("%02d",seconds)), totalHashFound);
			if(pps > 0)
				emit updateBruteStatus(3, pct, tr("Currently bruteforcing with %1 available nodes (%2 Mil/sec).").arg(successfulServerList.count()).arg(tmp.sprintf("%.2f",(double)pps)));
		}
		msleep(1000);
	}

	for(i=0; i < serverConList.count(); i++) {
		delete serverConList[i];
	}

	serverConList.clear();

	if(keysLeft <= 0) {
		emit updateBruteStatus(3, pct, tr("Bruteforcing  %1 available nodes."));
	}
	distributedServer->terminate();
}
